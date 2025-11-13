# -*- coding: utf-8 -*-
# PRM step → (optional) group-level standardization on steps → per-trajectory projection (optional) → suffix-sum on steps → broadcast to tokens
from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass
import torch
import math
from agentevolver.module.adv_processor.prompt import get_positive_mask, rescale_score

# =========================
# Hyper & small utilities
# =========================

@dataclass
class PRMHyper:
    """
    A dataclass to hold hyperparameters for the Policy Reward Model (PRM) process.

    Attributes:
        consistent_scale (float): Weight for consistent steps.
        pos_unconsistent_scale (float): Weight for bad steps in successful trajectories.
        neg_unconsistent_scale (float): Weight for good steps in failed trajectories.
        eps (float): Small value to prevent division by zero.
        do_batch_norm (bool): Whether to perform group-level z-score normalization.
        equal_trajectory_weight (bool): Whether to give equal weight to each trajectory.
        fix_base (float): Base magnitude for the fix scheme.
        alpha (float): Balance coefficient for PRM weights.
        orm_distribution (str): ORM distribution method, either "last_step" or "all_steps".
        enable_length_normalization (bool): Whether to enable length normalization.
    """
    # Weights: higher weight for consistent steps, lower weight for inconsistent steps (used for allocation)
    consistent_scale: float = 1.0
    pos_unconsistent_scale: float = 0.2   # Weight for BAD steps in successful trajectories
    neg_unconsistent_scale: float = 0.2   # Weight for GOOD steps in failed trajectories
    eps: float = 1e-8
    do_batch_norm: bool = True          # Whether to perform group-wise z-score (at step level, used by allocation/decouple)
    equal_trajectory_weight: bool = True  # True=equal weight per trajectory (GRPO); False=flatten all steps into one large sample (GSPO)
    fix_base: float = 0.2                 # Base magnitude for fix scheme (good=+base, bad=-base)
    alpha: float = 1.0                   # PRM weight balance coefficient
    orm_distribution: str = "last_step"   # ORM distribution method: "last_step" or "all_steps"
    enable_length_normalization: bool = False  # Whether to enable length normalization (divide by sqrt(K))

def _ensure_tensor(x, device, dtype=None):
    """
    Ensures that the input is converted to a tensor with the specified device and, optionally, a specific data type.

    Args:
        x: The input to be converted to a tensor.
        device (torch.device): The device to which the tensor should be moved.
        dtype (torch.dtype, optional): The desired data type of the tensor. Defaults to None.

    Returns:
        torch.Tensor: The input converted to a tensor with the specified device and data type.
    """
    if torch.is_tensor(x):
        t = x.to(device=device)
        if dtype is not None:
            t = t.to(dtype)
        return t
    return torch.as_tensor(x, device=device, dtype=dtype)

def _num_steps_from_step_ids(step_ids_row: torch.Tensor) -> int:
    if step_ids_row.numel() == 0:
        return 0
    valid = step_ids_row >= 0
    if not torch.any(valid):
        return 0
    return int(step_ids_row[valid].max().item()) + 1

def _align_flags(flags: List[bool], K: int, is_success: bool) -> List[bool]:
    """
    Aligns the length of the flags list to match the specified number of steps K.
    If the flags list is shorter than K, it pads the list with a default value.
    If the flags list is longer than K, it truncates the list to K elements.

    Args:
        flags (List[bool]): The list of boolean flags.
        K (int): The target length to align the flags list to.
        is_success (bool): Determines the default value for padding. True if is_success, False otherwise.

    Returns:
        List[bool]: The aligned list of boolean flags.
    """
    if len(flags) == K:
        return list(flags)
    default_flag = True if is_success else False
    if len(flags) < K:
        return list(flags) + [default_flag] * (K - len(flags))
    else:
        return list(flags[:K])

def _group_zscore_on_steps(
    step_rewards_raw: List[List[float]],
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """
    Standardizes the step rewards within each group by subtracting the mean and dividing by the standard deviation.

    Args:
        step_rewards_raw (List[List[float]]): A list of lists containing the raw step rewards for each trajectory.
        group_ids (torch.Tensor): A tensor indicating the group ID for each trajectory.
        hyper (PRMHyper): An object containing hyperparameters, including whether to perform batch normalization and
                          whether to use equal trajectory weighting.

    Returns:
        List[List[float]]: A list of lists containing the standardized step rewards for each trajectory.
    """
    if not hyper.do_batch_norm:
        return [list(r) for r in step_rewards_raw]

    B = len(step_rewards_raw)
    gids = group_ids.view(-1).tolist()
    g2idx: Dict[int, List[int]] = {}
    for i, g in enumerate(gids):
        g2idx.setdefault(int(g), []).append(i)

    step_rewards_std: List[List[float]] = [[] for _ in range(B)]
    eps = float(hyper.eps)

    for _, idxs in g2idx.items():
        if hyper.equal_trajectory_weight:
            # === Equal trajectory weight: mean of means first, then mean of variances ===
            n_traj = 0
            mu_acc = 0.0
            for i in idxs:
                ri = step_rewards_raw[i]
                if not ri:
                    continue
                n_traj += 1
                # Accumulate trajectory means (equal weight)
                mu_acc += (math.fsum(ri) / len(ri))
            if n_traj == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                mu_g = mu_acc / n_traj
                # Group variance = mean squared deviation around mu_g within trajectories, then equal-weighted average across trajectories
                second_moments_sum = 0.0
                for i in idxs:
                    ri = step_rewards_raw[i]
                    if not ri:
                        continue
                    second_moments_sum += (math.fsum((x - mu_g) * (x - mu_g) for x in ri) / len(ri))
                var_g = (second_moments_sum / n_traj) if n_traj > 0 else 0.0
                sd_g = math.sqrt(var_g + eps)
        else:
            # === Flatten: two-pass streaming statistics (to avoid huge overhead of converting flat lists and tensors) ===
            total_cnt = 0
            total_sum = 0.0
            # pass1: count total steps and sum across the group → mean
            for i in idxs:
                ri = step_rewards_raw[i]
                if not ri:
                    continue
                total_cnt += len(ri)
                total_sum += math.fsum(ri)

            if total_cnt == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                mu_g = total_sum / total_cnt
                # pass2: accumulate second-order deviations → population variance (aligned with unbiased=False)
                M2 = 0.0
                for i in idxs:
                    ri = step_rewards_raw[i]
                    if not ri:
                        continue
                    M2 += math.fsum((x - mu_g) * (x - mu_g) for x in ri)
                var = M2 / total_cnt
                sd = math.sqrt(var)
                sd_g = sd if sd >= eps else eps

        inv = 1.0 / (sd_g + 1e-12)
        for i in idxs:
            ri = step_rewards_raw[i]
            if not ri:
                step_rewards_std[i] = []
            else:
                step_rewards_std[i] = [float((x - mu_g) * inv) for x in ri]

    return step_rewards_std



def _build_allocation(
    orm_scores: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """
    Builds a step-level allocation of rewards for each trajectory based on ORM scores, step flags, and other parameters.
    Ensures that the total reward for each trajectory aligns with the ORM score's sign and applies group normalization to the rewards.

    Args:
        orm_scores (torch.Tensor): Complete ORM scores, shape (B,), used to determine the direction and strategy of reward allocation.
        step_flags (List[List[bool]]): GOOD/BAD flags at the step level for each trajectory.
        step_ids (torch.Tensor): Step identifiers, shape (B, L_resp).
        group_ids (torch.Tensor): Group identifiers for group-level normalization, shape (B,).
        hyper (PRMHyper): PRM hyperparameter configuration.

    Returns:
        List[List[float]]: Step-level advantage rewards for each trajectory, after group mean subtraction.
    """
    B = step_ids.size(0)

    # ---------- Utility functions ----------
    def _p95(vals):
        if not vals:
            return 0.0
        s = sorted(vals)
        k = int(round(0.95 * (len(s) - 1)))
        return float(s[k])

    mean_eps = getattr(hyper, "zscore_mean_tol", 0.05)  # Tolerance for group-wise mean
    std_tol  = getattr(hyper, "zscore_std_tol", 0.2)    # Allowed deviation of std from 1 => interval [1-std_tol, 1+std_tol]
    small_mag_threshold = getattr(hyper, "small_mag_threshold", 0.05)

    # ---- Stage 1: Generate raw PRM rewards (consistent weight allocation, per-trajectory reward sum = ORM sign) ----
    step_rewards_raw: List[List[float]] = []

    # Monitoring: weight ratios / degenerate count / pre-normalization consistency invariants
    unit_weights: List[float] = []
    pos_consistent_shares: List[float] = []
    neg_consistent_shares: List[float] = []
    degenerate_total_w_count = 0
    pre_norm_sign_agree_flags: List[float] = []

    # Majority consistency (based on PRM annotations)
    pos_major_good = pos_cnt = 0
    neg_major_bad  = neg_cnt = 0

    # Cache flags for subsequent r_norm GAP calculation
    flags_cache: List[List[bool]] = []

    for i in range(B):
        # Get the number of steps in the current trajectory
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            step_rewards_raw.append([]); flags_cache.append([]); continue

        # Determine trajectory type and weight allocation strategy based on ORM score sign
        raw_orm = float(orm_scores[i].item())
        is_success = bool(get_positive_mask(raw_orm, threshold=0.5))

        # Align flags
        flags_i = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success)
        flags_cache.append(flags_i)

        # Count GOOD/BAD steps
        n_g = sum(1 for f in flags_i if f)
        n_b = K - n_g

        # Consistent/inconsistent weights
        if is_success:
            # Successful trajectory: consistent steps (GOOD) have higher weight, inconsistent steps (BAD) have lower weight
            w_g, w_b = hyper.consistent_scale, hyper.pos_unconsistent_scale
            sgn = +1.0
        else:
            # Failed trajectory: consistent steps (BAD) have lower weight, inconsistent steps (GOOD) have higher weight
            w_g, w_b = hyper.neg_unconsistent_scale, hyper.consistent_scale
            sgn = -1.0

        # Weight normalization: ensure trajectory total reward equals ORM sign
        total_w = n_g * w_g + n_b * w_b
        if total_w <= hyper.eps:
            unit = 0.0
            degenerate_total_w_count += 1
        else:
            unit = 1.0 / total_w
        unit_weights.append(unit)

        # Trajectory raw rewards (sum == sgn or degenerate to 0)
        r_raw = [sgn * (w_g * unit) if f else sgn * (w_b * unit) for f in flags_i]
        step_rewards_raw.append([float(x) for x in r_raw])

        # Monitoring: consistent weight ratio (pos: GOOD consistent; neg: BAD consistent)
        if total_w > hyper.eps:
            if is_success:
                pos_consistent_shares.append((n_g * w_g) / total_w)
            else:
                neg_consistent_shares.append((n_b * w_b) / total_w)

        # Monitoring: pre-norm invariant (sum(r_raw) should be consistent with ORM sign)
        raw_sum = sum(r_raw)
        # is_raw_sum_positive = get_positive_mask(raw_sum, threshold=0.0)
        raw_orm_sign = 1.0 if is_success else -1.0
        pre_norm_sign_agree_flags.append(1.0 if (raw_sum * raw_orm_sign) > 0 else 0.0)

        # Majority consistency (PRM annotation vs ORM direction)
        is_good_majority = (n_g > n_b) # Add a readable boolean variable
        if is_success:
            pos_cnt += 1
            if is_good_majority:
                pos_major_good += 1
        else: # not is_success
            neg_cnt += 1
            if not is_good_majority: # not (n_g > n_b) is equivalent to n_b >= n_g
                neg_major_bad += 1


    # ---- Stage 2: Group-wise z-score normalization (to obtain the true advantage function) ----
    r_norm = _group_zscore_on_steps(step_rewards_raw, group_ids, hyper)

    # Monitoring: group-wise mean/variance (aggregate all steps by group)
    gid_list = group_ids.view(-1).tolist()
    group_vals: Dict[int, List[float]] = {}
    all_abs_rnorm: List[float] = []
    for i in range(B):
        g = int(gid_list[i])
        vals = r_norm[i]
        if not vals:
            continue
        group_vals.setdefault(g, []).extend(vals)
        all_abs_rnorm.extend(abs(x) for x in vals)

    group_mean_abs = []
    group_std = []
    zscore_bad_group_cnt = 0
    for g, vals in group_vals.items():
        t = torch.tensor(vals, dtype=torch.float32)
        m = float(t.mean().item())
        s = float(t.std(unbiased=False).item())
        group_mean_abs.append(abs(m))
        group_std.append(s)
        if (abs(m) > mean_eps) or (s < (1 - std_tol)) or (s > (1 + std_tol)):
            zscore_bad_group_cnt += 1

    r_norm_group_mean_abs_p95 = _p95(group_mean_abs) if group_mean_abs else 0.0
    r_norm_group_std_p95 = _p95(group_std) if group_std else 0.0

    # Monitoring: r_norm separability of GOOD/BAD (measured separately by ORM positive/negative)
    gap_pos_list = []
    gap_neg_list = []
    for i in range(B):
        vals = r_norm[i]
        if not vals:
            continue
        flags_i = flags_cache[i]
        raw_orm = float(orm_scores[i].item())
        good_vals = [v for v, f in zip(vals, flags_i) if f]
        bad_vals  = [v for v, f in zip(vals, flags_i) if not f]
        is_orm_positive_current = get_positive_mask(float(orm_scores[i].item()))

        if is_orm_positive_current:
            if good_vals and bad_vals:
                gap_pos_list.append(float(torch.tensor(good_vals).mean() - torch.tensor(bad_vals).mean()))
        else:
            if good_vals and bad_vals:
                gap_neg_list.append(float(torch.tensor(bad_vals).mean() - torch.tensor(good_vals).mean()))
    good_bad_rnorm_gap_pos = float(torch.tensor(gap_pos_list).mean().item()) if gap_pos_list else 0.0
    good_bad_rnorm_gap_neg = float(torch.tensor(gap_neg_list).mean().item()) if gap_neg_list else 0.0

    # Monitoring: small magnitude ratio (whether diluted)
    if all_abs_rnorm:
        rnorm_small_mag_ratio = float(sum(1 for x in all_abs_rnorm if x < small_mag_threshold) / len(all_abs_rnorm))
    else:
        rnorm_small_mag_ratio = 0.0

    # ---------- Stage 3: Group-wise normalize ORM and overlay on r_norm (allocation strategy consistent with decouple) ----------
    alpha = getattr(hyper, "alpha", 1.0)
    orm_distribution = getattr(hyper, "orm_distribution", "last_step")

    orm_list = orm_scores.detach().cpu().tolist()
    g2idx: Dict[int, List[int]] = {}
    for i, g in enumerate(gid_list):
        g2idx.setdefault(int(g), []).append(i)

    orm_scores_std = [0.0] * B
    for _, idxs in g2idx.items():
        group_vals_orm = [orm_list[i] for i in idxs]
        t = torch.tensor(group_vals_orm, dtype=torch.float32)
        m = t.mean()
        s = t.std(unbiased=False)
        if s <= hyper.eps:
            for i in idxs:
                orm_scores_std[i] = float(orm_list[i] - m.item())
        else:
            denom = s.item() + 1e-12
            for i in idxs:
                orm_scores_std[i] = float((orm_list[i] - m.item()) / denom)

    combined_rewards: List[List[float]] = []
    # Monitoring: ORM/PRM dominance & post-normalization consistency
    per_traj_attr_abs_sum = []
    per_traj_out_abs_sum  = []
    per_traj_out_last_abs = []
    sum_step_reward_sign_agree_flags: List[float] = []

    for i in range(B):
        steps_i = r_norm[i]
        if not steps_i:
            combined_rewards.append([]); continue
        K = len(steps_i)
        ostd = orm_scores_std[i]

        # Combination
        if orm_distribution == "last_step":
            arr = [alpha * x for x in steps_i]
            arr[-1] = arr[-1] + ostd
        elif orm_distribution == "all_steps":
            arr = [alpha * x + ostd for x in steps_i]
        else:
            raise ValueError(f"Unknown orm_distribution: {orm_distribution}")

        combined_rewards.append([float(v) for v in arr])

        # Monitoring: dominance (aligned with decouple)
        a_abs = sum(abs(alpha * x) for x in steps_i)          # α * Σ|r_norm|
        if orm_distribution == "last_step":
            o_abs = abs(ostd)                                 # Σ|ORM| (last_step mode)
            o_last = abs(ostd)
        else:
            o_abs = K * abs(ostd)                             # all_steps: each step has the same orm_std
            o_last = abs(ostd)

        per_traj_attr_abs_sum.append(float(a_abs))
        per_traj_out_abs_sum.append(float(o_abs))
        per_traj_out_last_abs.append(float(o_last))

        # Post-normalization consistency: ∑(combined_step_reward) vs original ORM sign
        is_orm_positive = get_positive_mask(float(orm_scores[i].item()), threshold=0.5)
        is_sum_positive = get_positive_mask(sum(arr), threshold=0.0)
        signs_agree = (is_sum_positive == is_orm_positive)
        sum_step_reward_sign_agree_flags.append(float(signs_agree))


    # outcome_share_last_mean & alpha_effective
    shares = []
    for a_abs, o_last in zip(per_traj_attr_abs_sum, per_traj_out_last_abs):
        denom = o_last + a_abs + 1e-12
        shares.append(float(o_last / denom))
    outcome_share_last_mean = float(sum(shares) / max(1, len(shares)))

    alpha_ratios = []
    for a_abs, o_abs in zip(per_traj_attr_abs_sum, per_traj_out_abs_sum):
        denom = o_abs + 1e-12
        alpha_ratios.append(float(a_abs / denom))
    alpha_effective = float(sum(alpha_ratios) / max(1, len(alpha_ratios)))

    sum_step_reward_sign_agree = float(sum(sum_step_reward_sign_agree_flags) / max(1, len(sum_step_reward_sign_agree_flags)))

    # post-norm invariant (after z-score, sum should be approximately 0)
    post_norm_sum_vals = []
    for vals in r_norm:
        if vals:
            post_norm_sum_vals.append(sum(vals))
    post_norm_sum_mean = float(torch.tensor(post_norm_sum_vals, dtype=torch.float32).mean().item()) if post_norm_sum_vals else 0.0

    # Majority consistency (aligned with decouple metrics for horizontal comparison)
    pos_rate = float(pos_major_good / max(1, pos_cnt))
    neg_rate = float(neg_major_bad  / max(1, neg_cnt))

    # ---------- Summary metrics ----------
    alloc_stats = {
        # §1 Whether weight allocation works as designed
        "prm_allocation/consistent_weight_share_pos": float(torch.tensor(pos_consistent_shares).mean().item()) if pos_consistent_shares else 0.0,
        "prm_allocation/consistent_weight_share_neg": float(torch.tensor(neg_consistent_shares).mean().item()) if neg_consistent_shares else 0.0,
        "prm_allocation/unit_weight_mean": float(torch.tensor(unit_weights).mean().item()) if unit_weights else 0.0,
        "prm_allocation/unit_weight_p95": _p95(unit_weights),
        "prm_allocation/degenerate_total_w_count": float(degenerate_total_w_count),

        # §2 z-score effectiveness
        "prm_allocation/r_norm_group_mean_abs_p95": r_norm_group_mean_abs_p95,
        "prm_allocation/r_norm_group_std_p95": r_norm_group_std_p95,
        "prm_allocation/zscore_bad_group_cnt": float(zscore_bad_group_cnt),

        # §3 Relationship between PRM annotations and r_norm
        "prm_allocation/good_bad_rnorm_gap_pos": good_bad_rnorm_gap_pos,
        "prm_allocation/good_bad_rnorm_gap_neg": good_bad_rnorm_gap_neg,
        "prm_allocation/rnorm_small_mag_ratio": rnorm_small_mag_ratio,

        # §4 Invariant checks
        "prm_allocation/pre_norm_sum_sign_agree": float(sum(pre_norm_sign_agree_flags) / max(1, len(pre_norm_sign_agree_flags))),
        "prm_allocation/post_norm_sum_mean": post_norm_sum_mean,

        # §6 Dominance and consistency (after overlaying ORM)
        "prm_allocation/outcome_share_last_mean": outcome_share_last_mean,
        "prm_allocation/alpha_effective": alpha_effective,
        "prm_allocation/sum_step_reward_sign_agree": sum_step_reward_sign_agree,

        # Majority consistency (aligned with decouple for horizontal comparison)
        "prm_allocation/pos_traj_prm_good_majority_rate": pos_rate,
        "prm_allocation/neg_traj_prm_bad_majority_rate": neg_rate,
    }

    return combined_rewards, alloc_stats


import math
from typing import List, Dict
import torch

def _build_decouple(
    orm_full_scores: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: "PRMHyper"
) -> List[List[float]]:
    """
    Decouples and standardizes PRM and ORM rewards separately before combining them.

    Args:
        orm_full_scores (torch.Tensor): Full ORM scores for each trajectory.
        step_flags (List[List[bool]]): Flags indicating success or failure for each step in each trajectory.
        step_ids (torch.Tensor): Step IDs for each trajectory.
        group_ids (torch.Tensor): Group IDs for each trajectory.
        hyper (PRMHyper): Hyperparameters for the decoupling process.

    Returns:
        List[List[float]]: Combined and standardized PRM and ORM rewards for each trajectory.
    """

    B = step_ids.size(0)
    alpha = hyper.alpha
    orm_distribution = hyper.orm_distribution
    enable_length_normalization = hyper.enable_length_normalization  # New parameter to control whether to apply sqrt length normalization

    # ---- 1. Construct base PRM rewards ----
    prm_rewards_raw: List[List[float]] = []
    for i in range(B):
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            prm_rewards_raw.append([])
            continue
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success=True)
        prm_rewards = [hyper.fix_base if f else -hyper.fix_base for f in flags]
        prm_rewards_raw.append(prm_rewards)

    # ---- 2. Perform z-score normalization on PRM rewards within groups ----
    prm_rewards_std = _group_zscore_on_steps(prm_rewards_raw, group_ids, hyper)

    # ---- 3. Perform group-wise normalization on ORM scores ----
    orm_scores = orm_full_scores.cpu().tolist()
    gids = group_ids.view(-1).tolist()
    g2idx: Dict[int, List[int]] = {}
    for i, g in enumerate(gids):
        g2idx.setdefault(int(g), []).append(i)

    orm_scores_std = [0.0] * B
    for _, idxs in g2idx.items():
        group_orms = [orm_scores[i] for i in idxs]
        if len(group_orms) == 0:
            continue
        orm_tensor = torch.tensor(group_orms, dtype=torch.float32)
        orm_mean = orm_tensor.mean()
        orm_std = orm_tensor.std(unbiased=False)
        if orm_std <= hyper.eps:
            for i in idxs:
                orm_scores_std[i] = float(orm_scores[i] - orm_mean.item())
        else:
            for i in idxs:
                orm_scores_std[i] = float((orm_scores[i] - orm_mean.item()) / (orm_std.item() + 1e-12))

    # ---- 4. Combine standardized PRM and ORM rewards ----
    combined_rewards: List[List[float]] = []

    # Prepare containers for statistics
    per_traj_attr_abs_sum = []   # Sum of α * |PRM_std| for each trajectory (excluding ORM)
    per_traj_out_abs_sum  = []   # Sum of ORM_std for each trajectory (all steps: K * |orm_std|; last step: |orm_std|)
    per_traj_out_last_abs = []   # Absolute value of ORM at the last step (for outcome_share_last_mean)
    sum_sign_agree_flags  = []   # Whether the sum of combined_step_reward agrees with the original ORM sign
    pos_major_good, pos_cnt = 0, 0
    neg_major_bad , neg_cnt = 0, 0

    # Prepare containers for PRM/ORM distribution statistics
    flat_attr_vals = []          # All step PRM standardized values (not multiplied by α)
    out_vals       = []          # One ORM standardized value per trajectory

    for i in range(B):
        if not prm_rewards_std[i]:
            combined_rewards.append([])
            continue

        prm_std = prm_rewards_std[i]
        orm_std = orm_scores_std[i]
        K = len(prm_std)
        # --- PRM/ORM distribution statistics sampling ---
        flat_attr_vals.extend(prm_std)
        out_vals.append(float(orm_std))

        # 🔥 Key difference: whether to calculate length normalization factor
        if enable_length_normalization:
            length_scale = 1.0 / math.sqrt(max(K, 1))
            print(f"Trajectory {i}: length={K}, length scale factor=1/sqrt({K})={length_scale:.4f}")
        else:
            length_scale = 1.0
            print(f"Trajectory {i}: length={K}, no length normalization (scale factor=1.0)")

        combined = []
        # Construct combined_step_reward step by step, and calculate various sums for per-traj
        attr_abs_sum = 0.0  # α * Σ_j |prm_std[j]|
        for j, prm_reward in enumerate(prm_std):
            if orm_distribution == "last_step":
                if j == K - 1:
                    combined_reward = alpha * prm_reward + orm_std
                else:
                    combined_reward = alpha * prm_reward
            elif orm_distribution == "all_steps":
                combined_reward = alpha * prm_reward + orm_std
            elif orm_distribution == "only_prm":
                combined_reward = prm_reward
            else:
                raise ValueError(f"Unknown orm_distribution: {orm_distribution}")

            final_reward = combined_reward * length_scale
            combined.append(float(final_reward))
            attr_abs_sum += abs(alpha * prm_reward)

        # Absolute contribution of ORM (per trajectory)
        if orm_distribution == "last_step":
            out_abs_sum = abs(orm_std)               # Only added at the last step
            out_last_abs = abs(orm_std)
        else:  # "all_steps"
            out_abs_sum = K * abs(orm_std)           # all_steps: each step has the same orm_std
            out_last_abs = abs(orm_std)

        per_traj_attr_abs_sum.append(float(attr_abs_sum))
        per_traj_out_abs_sum.append(float(out_abs_sum))
        per_traj_out_last_abs.append(float(out_last_abs))

        # ∑(combined_step_reward) consistency with the "original" ORM sign (not using z-score sign)
        is_orm_positive = get_positive_mask(float(orm_full_scores[i].item()), threshold=0.5)
        combined_sum = sum(combined)
        is_sum_positive = get_positive_mask(combined_sum, threshold=0.0)
        signs_agree = (is_sum_positive == is_orm_positive)
        sum_sign_agree_flags.append(float(signs_agree))

        # "Majority" consistency of PRM annotations in positive/negative trajectories
        flags_i = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success=is_orm_positive)
        n_g = sum(1 for f in flags_i if f)
        n_b = K - n_g
        is_good_majority = (n_g > n_b) # Add a readable variable
        if is_orm_positive:
            pos_cnt += 1
            if is_good_majority:
                pos_major_good += 1
        else:
            neg_cnt += 1
            if not is_good_majority:
                neg_major_bad += 1

        combined_rewards.append(combined)

    # === Decouple statistics ===
    # 1) mean/std of PRM/ORM normalized distribution
    if len(flat_attr_vals) == 0:
        attr_mean, attr_std = 0.0, 0.0
    else:
        t_attr = torch.tensor(flat_attr_vals, dtype=torch.float32)
        attr_mean = float(t_attr.mean().item())
        attr_std  = float(t_attr.std(unbiased=False).item())

    if len(out_vals) == 0:
        out_mean, out_std = 0.0, 0.0
    else:
        t_out = torch.tensor(out_vals, dtype=torch.float32)
        out_mean = float(t_out.mean().item())
        out_std  = float(t_out.std(unbiased=False).item())

    # 2) outcome_share_last_mean: |ORM(last step)| / (|ORM(last step)| + α * Σ|PRM_std|)
    shares = []
    for a_abs, o_last in zip(per_traj_attr_abs_sum, per_traj_out_last_abs):
        denom = o_last + a_abs + 1e-12
        shares.append(float(o_last / denom))
    outcome_share_last_mean = float(sum(shares) / max(1, len(shares)))

    # 3) alpha_effective: α * Σ|PRM_std| / (Σ|ORM|), calculate ratio per trajectory then average
    alpha_ratios = []
    for a_abs, o_abs, i in zip(per_traj_attr_abs_sum, per_traj_out_abs_sum, range(len(per_traj_out_abs_sum))):
        denom = o_abs + 1e-12
        alpha_ratios.append(float(a_abs / denom))
    alpha_effective = float(sum(alpha_ratios) / max(1, len(alpha_ratios)))

    # 4) Proportion of ∑(combined_step_reward) consistent with the original ORM sign
    sum_step_reward_sign_agree = float(sum(sum_sign_agree_flags) / max(1, len(sum_sign_agree_flags)))

    # 5) "Global consistency" of PRM annotations with ORM (majority)
    pos_rate = float(pos_major_good / max(1, pos_cnt))
    neg_rate = float(neg_major_bad  / max(1, neg_cnt))

    decouple_stats = {
        "prm/decouple/attr_mean": attr_mean,
        "prm/decouple/attr_std": attr_std,
        "prm/decouple/out_mean": out_mean,
        "prm/decouple/out_std": out_std,
        "prm/decouple/outcome_share_last_mean": outcome_share_last_mean,
        "prm/decouple/alpha_effective": alpha_effective,
        "prm/decouple/sum_step_reward_sign_agree": sum_step_reward_sign_agree,
        "prm/decouple/pos_traj_prm_good_majority_rate": pos_rate,
        "prm/decouple/neg_traj_prm_bad_majority_rate": neg_rate,
    }

    # Note: Return (rewards, stats) tuple (only decouple does this), other schemes still return rewards only
    return combined_rewards, decouple_stats
# =========================
# Step → Token broadcast + suffix-sum
# =========================

def suffix_sum_on_steps(step_rewards: List[List[float]]) -> List[List[float]]:
    """
    Computes the suffix sum (cumulative sum from the end to the beginning) for each trajectory's step rewards.

    Args:
        step_rewards (List[List[float]]): A list of lists, where each inner list contains the step rewards for a single trajectory.

    Returns:
        List[List[float]]: A list of lists, where each inner list contains the suffix sums of the step rewards for a single trajectory.
    """
    adv: List[List[float]] = []
    for r in step_rewards:
        if not r:
            adv.append([]); continue
        t = torch.tensor(r, dtype=torch.float32)
        s = torch.flip(torch.cumsum(torch.flip(t, dims=[0]), dim=0), dims=[0])
        adv.append([float(x) for x in s])
    return adv

def broadcast_step_adv_to_tokens(
    step_adv: List[List[float]],
    step_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Broadcasts step-level advantage values to the token level.

    This function assigns the advantage value of each step to the corresponding token positions
    based on the provided `step_ids`. Non-response tokens (indicated by -1 in `step_ids`) are
    kept at 0.

    Args:
        step_adv (List[List[float]]): A list of step advantage values for each trajectory.
        step_ids (torch.Tensor): A tensor of step identifiers, shape (B, L_resp), where -1
                                 indicates non-response tokens.

    Returns:
        torch.Tensor: A tensor of token-level advantage values, shape (B, L_resp).
    """
    device = step_ids.device
    B, L = step_ids.shape
    out = torch.zeros((B, L), device=device, dtype=torch.float32)
    for i in range(B):
        if not step_adv[i]:
            continue
        adv_i = torch.tensor(step_adv[i], device=device, dtype=torch.float32)
        sid_row = step_ids[i]
        valid = sid_row >= 0
        if torch.any(valid):
            sids = sid_row[valid]
            out[i, valid] = adv_i[sids]
    return out

# =========================
# Entry
# =========================

def compute_prm_grpo_advantages(
    batch,                          # DataProto or compatible structure: batch.batch[...] can be indexed
    step_flags: List[List[bool]],   # GOOD/BAD flags for each trajectory
    hyper: Optional[PRMHyper] = None,
    scheme: str = "decouple",   #  "allocation" | "decouple"
) -> dict:
    """
    Computes the PRM-GRPO advantages for a given batch of data.

    The function follows these steps:
    1. Data preparation: Extract necessary fields and compute ORM scores.
    2. Scheme selection: Choose the reward construction scheme and build step-level rewards.
    3. Advantage calculation: Compute step-level advantages and broadcast to token-level.
    4. Result return: Return token-level advantages and original ORM scores.

    Args:
        batch: Data batch containing responses, step_ids, group_ids, and token_level_rewards.
            - responses: Response tensor.
            - step_ids: Step identifiers, shape (B, L_resp), -1 for non-response tokens.
            - group_ids: Group identifiers for grouping, shape (B,).
            - token_level_rewards: Token-level rewards for ORM score computation.
        step_flags: Step-level GOOD/BAD flags for each trajectory.
        hyper: PRM hyperparameters, uses default if None.
        scheme: Reward construction scheme, either "allocation" or "decouple".

    Returns:
        dict: A dictionary containing:
            - advantages: (B, L_resp) token-level advantages.
            - orm_scalar: (B,) ORM scores for each trajectory.
    """
    if hyper is None:
        hyper = PRMHyper()

    # ---- 1. Data preparation stage: extract necessary fields ----
    # Get device information to ensure all tensors are on the same device
    responses = batch.batch["responses"]
    device = responses.device if torch.is_tensor(responses) else torch.as_tensor(responses).device

    # Extract step_ids and group_ids, ensuring correct data types
    step_ids = _ensure_tensor(batch.batch["step_ids"], device=device, dtype=torch.long)      # (B, L_resp) with -1 for non-response
    # >>> add begin: align to actual response length <<<
    target_L = responses.size(1)
    if step_ids.size(1) != target_L:
        if step_ids.size(1) > target_L:
            step_ids = step_ids[:, :target_L]
        else:
            pad = torch.full(
                (step_ids.size(0), target_L - step_ids.size(1)),
                -1, device=step_ids.device, dtype=step_ids.dtype
            )
            step_ids = torch.cat([step_ids, pad], dim=1)
    # <<< add end
    group_ids = _ensure_tensor(batch.batch["group_ids"], device=device, dtype=torch.long).view(-1)

    # ---- 2. Extract token-level rewards ----
    # Try multiple possible field names to get token-level rewards
    token_keys_try = ["token_level_rewards", "response_token_level_rewards", "token_rewards"]
    token_level_rewards = None
    for k in token_keys_try:
        if k in batch.batch:
            token_level_rewards = _ensure_tensor(batch.batch[k], device=device, dtype=torch.float32)
            break
    if token_level_rewards is None:
        raise KeyError("token-level rewards not found in batch (tried keys: token_level_rewards / response_token_level_rewards / token_rewards)")

    # ---- 3. ORM processing: calculate ORM scores ----
    # Sum token-level rewards to get trajectory-level ORM scores for reward construction in various schemes
    orm_scores = token_level_rewards.sum(dim=1)  

    # ---- 4. Scheme selection stage: select specific reward construction scheme based on scheme parameter ----
    extra_metrics = {}
    scheme = (scheme or "decouple").lower()
    step_rewards = None
    if scheme == "allocation":
        # Scheme 2: allocation —— consistent weight allocation + group-wise mean subtraction centering
        step_rewards, extra_metrics = _build_allocation(orm_scores, step_flags, step_ids, group_ids, hyper)
    elif scheme == "decouple":
        # Scheme 4: decouple —— PRM and ORM separately normalized then combined
        step_rewards, extra_metrics = _build_decouple(orm_scores, step_flags, step_ids, group_ids, hyper,)
    else:
        raise ValueError(f"Unknown PRM scheme: {scheme} (expected one of: allocation | decouple)")

    # ---- 5. Advantage calculation stage: step suffix-sum + broadcast to tokens ----
    # Perform suffix-sum on step-level rewards to get step-level advantage values
    step_adv = suffix_sum_on_steps(step_rewards)
    advantages = broadcast_step_adv_to_tokens(step_adv, step_ids)

    # ---- 6. Result return stage: construct return dictionary ----
    # Return token-level advantage values and original ORM scores
    return {
        "advantages": advantages,        # (B, L_resp) token-level advantage values
        "orm_scores": orm_scores,         # (B,) ±1 for each trajectory
        "metrics":  extra_metrics,      # ✅ Only decouple has this
    }
