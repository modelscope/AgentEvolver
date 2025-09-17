# -*- coding: utf-8 -*-
# PRM step → (optional) group-level standardization on steps → per-trajectory projection (optional) → suffix-sum on steps → broadcast to tokens
from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass
import torch
import math

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
    # 权重：一致性步的权重大，不一致性步的权重小（用于 allocation）
    consistent_scale: float = 1.0
    pos_unconsistent_scale: float = 0.2   # 成功轨迹里的 BAD 步权重
    neg_unconsistent_scale: float = 0.2   # 失败轨迹里的 GOOD 步权重
    eps: float = 1e-8
    do_batch_norm: bool = True          # 是否做组内 z-score（按 step 级，allocation/decouple 会用到）
    equal_trajectory_weight: bool = True  # True=每条轨迹等权（GRPO）；False=把所有 step 拉平成一个大样本（GSPO）
    fix_base: float = 0.2                 # fix 方案的基础幅度（good=+base, bad=-base）
    alpha: float = 1.0                   # PRM权重平衡系数
    orm_distribution: str = "last_step"   # ORM分配方式："last_step" 或 "all_steps"
    enable_length_normalization: bool = False  # 是否启用长度正则化（除以sqrt(K)）

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
        t = x.to(device=device)  # ⭐ Convert the tensor to the specified device
        if dtype is not None:
            t = t.to(dtype)
        return t
    return torch.as_tensor(x, device=device, dtype=dtype)  # ⭐ Convert the input to a tensor

def _num_steps_from_step_ids(step_ids_row: torch.Tensor) -> int:
    """
    Calculates the number of steps in a trajectory from the given step IDs.

    Args:
        step_ids_row (torch.Tensor): A tensor containing step IDs.

    Returns:
        int: The number of steps in the trajectory.
    """
    if step_ids_row.numel() == 0:
        return 0
    m = torch.amax(step_ids_row)  # ⭐ Find the maximum value in the step_ids_row tensor
    return int(m.item() + 1) if m.item() >= 0 else 0  # ⭐ Convert the maximum value to an integer and add 1 to get the total number of steps

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
        return list(flags) + [default_flag] * (K - len(flags))  # ⭐ Pad the list with the default value
    else:
        return list(flags[:K])  # ⭐ Truncate the list to K elements

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
            # === 轨迹等权：先均值的均值，再均方差的均值 ===
            n_traj = 0
            mu_acc = 0.0
            for i in idxs:
                ri = step_rewards_raw[i]
                if not ri:
                    continue
                n_traj += 1
                # 轨迹均值累加（等权）
                mu_acc += (math.fsum(ri) / len(ri))
            if n_traj == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                mu_g = mu_acc / n_traj
                # 组方差 = 轨迹内围绕 mu_g 的均方差，再对轨迹做等权平均
                second_moments_sum = 0.0
                for i in idxs:
                    ri = step_rewards_raw[i]
                    if not ri:
                        continue
                    second_moments_sum += (math.fsum((x - mu_g) * (x - mu_g) for x in ri) / len(ri))
                var_g = (second_moments_sum / n_traj) if n_traj > 0 else 0.0
                sd_g = math.sqrt(var_g + eps)
        else:
            # === 拉平：两遍流式统计（避免 flat 列表与 tensor 转换的巨大开销）===
            total_cnt = 0
            total_sum = 0.0
            # pass1: 统计全组总步数与总和 → 均值
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
                # pass2: 累加二阶偏差 → population variance（与 unbiased=False 对齐）
                M2 = 0.0
                for i in idxs:
                    ri = step_rewards_raw[i]
                    if not ri:
                        continue
                    M2 += math.fsum((x - mu_g) * (x - mu_g) for x in ri)
                var = M2 / total_cnt
                sd = math.sqrt(var)
                sd_g = sd if sd >= eps else eps

        inv = 1.0 / (sd_g + 1e-12)  # ⭐ Calculate the inverse of the standard deviation to avoid division by zero
        for i in idxs:
            ri = step_rewards_raw[i]
            if not ri:
                step_rewards_std[i] = []
            else:
                # 与原逻辑一致：按组统计量逐步标准化
                step_rewards_std[i] = [float((x - mu_g) * inv) for x in ri]  # ⭐ Standardize the step rewards using the group mean and inverse standard deviation

    return step_rewards_std


def _per_traj_scale_to_target_sum(
    r_std: List[float],
    target_sum: float,
    eps: float,
) -> List[float]:
    """
    Scales the step rewards of a trajectory so that their sum equals a target value.
    If the current sum of the rewards is close to zero, it evenly distributes the target sum across all steps.

    Args:
        r_std (List[float]): The list of standardized step rewards.
        target_sum (float): The target sum value.
        eps (float): A small constant for numerical stability.

    Returns:
        List[float]: The scaled step rewards.
    """
    if len(r_std) == 0:
        return []
    cur = sum(r_std)
    if abs(cur) <= eps:
        return [target_sum / len(r_std) for _ in r_std]  # ⭐ Evenly distribute the target sum when the current sum is close to zero
    scale = target_sum / cur  # ⭐ Calculate the scaling factor to reach the target sum
    return [float(x * scale) for x in r_std]  # ⭐ Apply the scaling factor to each reward
def _build_fix(
    orm_scores: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """
    Constructs step-level rewards for trajectories with a fixed base reward and adjusts the last step's reward based on the ORM score's sign.
    Additionally, it performs group-level z-score normalization on the constructed rewards.

    Args:
        orm_scores (torch.Tensor): Complete ORM scores, shape (B,), used to determine the reward direction.
        step_flags (List[List[bool]]): Step-level GOOD/BAD flags for each trajectory.
        step_ids (torch.Tensor): Step identifiers, shape (B, L_resp), -1 indicates non-response tokens.
        group_ids (torch.Tensor): Group identifiers for group-wise normalization, shape (B,).
        hyper (PRMHyper): PRM hyperparameter configuration, primarily using the fix_base parameter.

    Returns:
        List[List[float]]: A list of step-level rewards for each trajectory, with lengths matching the number of steps.

    Example:
        orm_scores = [2.5, -1.5]  # First trajectory is successful, second is not
        step_flags = [[True, False, True], [False, True]]  # Step flags for two trajectories
        hyper.fix_base = 0.2
        # Output example:
        # [[0.2, -0.2, 0.2],  # First trajectory: +0.2-0.2+0.2+1.0 = 1.2
        #  [-0.2, 0.2]]       # Second trajectory: -0.2+0.2-1.0 = -1.0
    """
    B = step_ids.size(0)
    prm_rewards_raw: List[List[float]] = []
    base = float(hyper.fix_base)

    # ---- 1. 构造原始 PRM 奖励 ----
    for i in range(B):
        # 获取当前轨迹的step数量
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            prm_rewards_raw.append([]); continue

        # 对齐step flags长度，确保与step数量一致
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success=True)

        # 构造基础PRM奖励：GOOD步骤为+base，BAD步骤为-base
        r = [(+base if f else -base) for f in flags]

        # 基于ORM分数符号调整最后一步奖励，确保整体奖励方向与ORM一致
        orm_sign = 1.0 if float(orm_scores[i].item()) > 0 else -1.0
        if len(r) > 0:
            r[-1] += orm_sign  # ⭐ Adjust the last step's reward based on the ORM score's sign

        prm_rewards_raw.append(r)

    # ---- 2. 组内 z-score (标准化) ----
    # 使用 _group_zscore_on_steps 来做组内标准化
    prm_rewards_norm = _group_zscore_on_steps(prm_rewards_raw, group_ids, hyper)
    return prm_rewards_norm

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

    # ---------- 工具 ----------
    def _p95(vals):
        if not vals:
            return 0.0
        s = sorted(vals)
        k = int(round(0.95 * (len(s) - 1)))
        return float(s[k])

    mean_eps = getattr(hyper, "zscore_mean_tol", 0.05)  # 组内均值容差
    std_tol  = getattr(hyper, "zscore_std_tol", 0.2)    # std 允许偏离 1 的幅度 => 区间 [1-std_tol, 1+std_tol]
    small_mag_threshold = getattr(hyper, "small_mag_threshold", 0.05)

    # ---- 第一阶段：生成原始PRM奖励（一致性权重瓜分，逐轨迹奖励和 = ORM符号）----
    step_rewards_raw: List[List[float]] = []

    # 监控：权重占比 / 退化计数 / 前置一致性不变量
    unit_weights: List[float] = []
    pos_consistent_shares: List[float] = []
    neg_consistent_shares: List[float] = []
    degenerate_total_w_count = 0
    pre_norm_sign_agree_flags: List[float] = []

    # 多数派一致性（基于 PRM 标注）
    pos_major_good = pos_cnt = 0
    neg_major_bad  = neg_cnt = 0

    # 记录 flags 供后续 r_norm 计算 GAP
    flags_cache: List[List[bool]] = []

    for i in range(B):
        # 获取当前轨迹的step数量
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            step_rewards_raw.append([]); flags_cache.append([]); continue

        # 根据ORM分数符号确定轨迹类型和权重分配策略
        raw_orm = float(orm_scores[i].item())
        is_success = bool(raw_orm > 0)

        # 对齐 flags
        flags_i = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success)
        flags_cache.append(flags_i)

        # GOOD/BAD 数
        n_g = sum(1 for f in flags_i if f)
        n_b = K - n_g

        # 一致/不一致权重
        if is_success:
            # 成功轨迹：一致性步骤(GOOD)权重高，不一致性步骤(BAD)权重低
            w_g, w_b = hyper.consistent_scale, hyper.pos_unconsistent_scale
            sgn = +1.0
        else:
            # 失败轨迹：一致性步骤(BAD)权重低，不一致性步骤(GOOD)权重高
            w_g, w_b = hyper.neg_unconsistent_scale, hyper.consistent_scale
            sgn = -1.0

        # 权重归一化：确保轨迹总奖励等于ORM符号
        total_w = n_g * w_g + n_b * w_b
        if total_w <= hyper.eps:
            unit = 0.0
            degenerate_total_w_count += 1
        else:
            unit = 1.0 / total_w  # ⭐ Normalize the weights to ensure the total reward equals the ORM sign
        unit_weights.append(unit)

        # 轨迹 raw 奖励（sum == sgn 或退化为 0）
        r_raw = [sgn * (w_g * unit) if f else sgn * (w_b * unit) for f in flags_i]
        step_rewards_raw.append([float(x) for x in r_raw])

        # 监控：一致性权重占比（pos: GOOD 一致；neg: BAD 一致）
        if total_w > hyper.eps:
            if is_success:
                pos_consistent_shares.append((n_g * w_g) / total_w)
            else:
                neg_consistent_shares.append((n_b * w_b) / total_w)

        # 监控：pre-norm 不变量（sum(r_raw) 与 ORM 符号应一致）
        raw_sum = sum(r_raw)
        raw_orm_sign = 1.0 if raw_orm > 0 else -1.0
        pre_norm_sign_agree_flags.append(1.0 if (raw_sum * raw_orm_sign) > 0 else 0.0)

        # 多数派一致性（PRM 标注 vs ORM 方向）
        if raw_orm > 0:
            pos_cnt += 1
            if n_g > n_b:
                pos_major_good += 1
        else:
            neg_cnt += 1
            if n_b >= n_g:
                neg_major_bad += 1

    # ---- 第二阶段：组内 z-score 标准化（获得真正的优势函数）----
    # 使用 _group_zscore_on_steps 函数进行标准化
    r_norm = _group_zscore_on_steps(step_rewards_raw, group_ids, hyper)  # ⭐ Perform group-wise z-score normalization on step rewards

    # 监控：组内均值/方差（按 group 聚合所有 step）
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

    # 监控：GOOD/BAD 的 r_norm 可分性（按 ORM 正负分别度量）
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
        if raw_orm > 0:
            if good_vals and bad_vals:
                gap_pos_list.append(float(torch.tensor(good_vals).mean() - torch.tensor(bad_vals).mean()))
        else:
            if good_vals and bad_vals:
                gap_neg_list.append(float(torch.tensor(bad_vals).mean() - torch.tensor(good_vals).mean()))
    good_bad_rnorm_gap_pos = float(torch.tensor(gap_pos_list).mean().item()) if gap_pos_list else 0.0
    good_bad_rnorm_gap_neg = float(torch.tensor(gap_neg_list).mean().item()) if gap_neg_list else 0.0

    # 监控：小幅度比例（是否被稀释）
    if all_abs_rnorm:
        rnorm_small_mag_ratio = float(sum(1 for x in all_abs_rnorm if x < small_mag_threshold) / len(all_abs_rnorm))
    else:
        rnorm_small_mag_ratio = 0.0

    # ---------- 第三阶段：组内标准化 ORM 并叠加到 r_norm（与 decouple 一致的分配策略） ----------
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
    # 监控：ORM/PRM 主导度 & 后置一致性
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

        # 组合
        if orm_distribution == "last_step":
            arr = [alpha * x for x in steps_i]
            arr[-1] = arr[-1] + ostd
        elif orm_distribution == "all_steps":
            arr = [alpha * x + ostd for x in steps_i]
        else:
            raise ValueError(f"Unknown orm_distribution: {orm_distribution}")

        combined_rewards.append([float(v) for v in arr])

        # 监控：主导度（与 decouple 对齐）
        a_abs = sum(abs(alpha * x) for x in steps_i)          # α * Σ|r_norm|
        if orm_distribution == "last_step":
            o_abs = abs(ostd)                                 # Σ|ORM|（last_step 模式）
            o_last = abs(ostd)
        else:
            o_abs = K * abs(ostd)                             # all_steps：每步都有同一 orm_std
            o_last = abs(ostd)

        per_traj_attr_abs_sum.append(float(a_abs))
        per_traj_out_abs_sum.append(float(o_abs))
        per_traj_out_last_abs.append(float(o_last))

        # 后置一致性：∑(combined_step_reward) vs 原始 ORM 符号
        raw_orm_sign = 1.0 if float(orm_scores[i].item()) > 0.0 else -1.0
        if sum(arr) * raw_orm_sign > 0:
            sum_step_reward_sign_agree_flags.append(1.0)
        else:
            sum_step_reward_sign_agree_flags.append(0.0)

    # outcome_share_last_mean & alpha_effective
    shares = []
    for a_abs, o_last in zip(per_traj_attr_abs_sum, per_traj_out_last_abs):
        denom = o_last + a_abs + 1e-12
        shares.append(float(o_last / denom))  # ⭐ Calculate the share of the last step's ORM score
    outcome_share_last_mean = float(sum(shares) / max(1, len(shares)))

    alpha_ratios = []
    for a_abs, o_abs in zip(per_traj_attr_abs_sum, per_traj_out_abs_sum):
        denom = o_abs + 1e-12
        alpha_ratios.append(float(a_abs / denom))  # ⭐ Calculate the ratio of attribute absolute sum to ORM absolute sum
    alpha_effective = float(sum(alpha_ratios) / max(1, len(alpha_ratios)))

    sum_step_reward_sign_agree = float(sum(sum_step_reward_sign_agree_flags) / max(1, len(sum_step_reward_sign_agree_flags)))

    # post-norm 不变量（z-score 后按理 sum≈0）
    post_norm_sum_vals = []
    for vals in r_norm:
        if vals:
            post_norm_sum_vals.append(sum(vals))
    post_norm_sum_mean = float(torch.tensor(post_norm_sum_vals, dtype=torch.float32).mean().item()) if post_norm_sum_vals else 0.0

    # 多数派一致性（与 decouple 指标对齐，便于横向比较）
    pos_rate = float(pos_major_good / max(1, pos_cnt))
    neg_rate = float(neg_major_bad  / max(1, neg_cnt))

    # ---------- 汇总指标 ----------
    alloc_stats = {
        # §1 权重分配是否按设计工作
        "prm_allocation/consistent_weight_share_pos": float(torch.tensor(pos_consistent_shares).mean().item()) if pos_consistent_shares else 0.0,
        "prm_allocation/consistent_weight_share_neg": float(torch.tensor(neg_consistent_shares).mean().item()) if neg_consistent_shares else 0.0,
        "prm_allocation/unit_weight_mean": float(torch.tensor(unit_weights).mean().item()) if unit_weights else 0.0,
        "prm_allocation/unit_weight_p95": _p95(unit_weights),
        "prm_allocation/degenerate_total_w_count": float(degenerate_total_w_count),

        # §2 z-score 有效性
        "prm_allocation/r_norm_group_mean_abs_p95": r_norm_group_mean_abs_p95,
        "prm_allocation/r_norm_group_std_p95": r_norm_group_std_p95,
        "prm_allocation/zscore_bad_group_cnt": float(zscore_bad_group_cnt),

        # §3 PRM 标注与 r_norm 的关系
        "prm_allocation/good_bad_rnorm_gap_pos": good_bad_rnorm_gap_pos,
        "prm_allocation/good_bad_rnorm_gap_neg": good_bad_rnorm_gap_neg,
        "prm_allocation/rnorm_small_mag_ratio": rnorm_small_mag_ratio,

        # §4 不变量检查
        "prm_allocation/pre_norm_sum_sign_agree": float(sum(pre_norm_sign_agree_flags) / max(1, len(pre_norm_sign_agree_flags))),
        "prm_allocation/post_norm_sum_mean": post_norm_sum_mean,

        # §6 （叠加 ORM 后的）主导度与一致性
        "prm_allocation/outcome_share_last_mean": outcome_share_last_mean,
        "prm_allocation/alpha_effective": alpha_effective,
        "prm_allocation/sum_step_reward_sign_agree": sum_step_reward_sign_agree,

        # 多数派一致性（和 decouple 对齐，便于横向比较）
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

        # ⭐ Combine standardized PRM and ORM rewards for each trajectory
        combined_rewards_i = [alpha * prm + orm for prm, orm in zip(prm_rewards_std[i], [orm_scores_std[i]] * len(prm_rewards_std[i]))]
        if enable_length_normalization:
            K = len(combined_rewards_i)
            combined_rewards_i = [r / math.sqrt(K) for r in combined_rewards_i]

        combined_rewards.append(combined_rewards_i)

        prm_std = prm_rewards_std[i]
        orm_std = orm_scores_std[i]
        K = len(prm_std)
        # --- PRM/ORM 分布统计采样 ---
        flat_attr_vals.extend(prm_std)
        out_vals.append(float(orm_std))

        # 🔥 关键区别：是否计算长度正则化因子
        if enable_length_normalization:
            length_scale = 1.0 / math.sqrt(max(K, 1))
            print(f"轨迹 {i}: 长度={K}, 长度缩放因子=1/sqrt({K})={length_scale:.4f}")
        else:
            length_scale = 1.0
            print(f"轨迹 {i}: 长度={K}, 无长度正则化 (缩放因子=1.0)")

        combined = []
        # 逐步构造 combined_step_reward，并计算 per-traj 的各种和
        attr_abs_sum = 0.0  # α * Σ_j |prm_std[j]|
        for j, prm_reward in enumerate(prm_std):
            if orm_distribution == "last_step":
                if j == K - 1:
                    combined_reward = alpha * prm_reward + orm_std
                else:
                    combined_reward = alpha * prm_reward
            elif orm_distribution == "all_steps":
                combined_reward = alpha * prm_reward + orm_std
            else:
                raise ValueError(f"Unknown orm_distribution: {orm_distribution}")

            final_reward = combined_reward * length_scale
            combined.append(float(final_reward))
            attr_abs_sum += abs(alpha * prm_reward)

        # ORM 的绝对贡献（逐轨迹）
        if orm_distribution == "last_step":
            out_abs_sum = abs(orm_std)               # 只在最后一步加
            out_last_abs = abs(orm_std)
        else:  # "all_steps"
            out_abs_sum = K * abs(orm_std)           # 每步都加同一个 orm_std
            out_last_abs = abs(orm_std)

        per_traj_attr_abs_sum.append(float(attr_abs_sum))
        per_traj_out_abs_sum.append(float(out_abs_sum))
        per_traj_out_last_abs.append(float(out_last_abs))

        # ∑(combined_step_reward) 与「原始」ORM 符号一致性（不使用 z-score 后的符号）
        combined_sum = sum(combined)
        raw_orm_sign = 1.0 if float(orm_full_scores[i].item()) > 0.0 else -1.0
        sum_sign_agree_flags.append(1.0 if (combined_sum * raw_orm_sign) > 0 else 0.0)

        # PRM 标注在正/负轨迹中的“多数派”一致性
        flags_i = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success=True)
        n_g = sum(1 for f in flags_i if f)
        n_b = K - n_g
        if raw_orm_sign > 0:
            pos_cnt += 1
            if n_g > n_b:
                pos_major_good += 1
        else:
            neg_cnt += 1
            if n_b >= n_g:
                neg_major_bad += 1

        combined_rewards.append(combined)

    # === Decouple 统计指标 ===
    # 1) PRM/ORM 标准化后分布的 mean/std
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

    # 2) outcome_share_last_mean：|ORM(最后一步)| / (|ORM(最后一步)| + α * Σ|PRM_std|)
    shares = []
    for a_abs, o_last in zip(per_traj_attr_abs_sum, per_traj_out_last_abs):
        denom = o_last + a_abs + 1e-12
        shares.append(float(o_last / denom))  # ⭐ 计算每条轨迹中 ORM 最后一步的贡献比例
    outcome_share_last_mean = float(sum(shares) / max(1, len(shares)))

    # 3) alpha_effective：α * Σ|PRM_std| / (Σ|ORM|)，按轨迹求比再做均值
    alpha_ratios = []
    for a_abs, o_abs, i in zip(per_traj_attr_abs_sum, per_traj_out_abs_sum, range(len(per_traj_out_abs_sum))):
        denom = o_abs + 1e-12
        alpha_ratios.append(float(a_abs / denom))  # ⭐ 计算每条轨迹中 PRM 和 ORM 的相对贡献
    alpha_effective = float(sum(alpha_ratios) / max(1, len(alpha_ratios)))

    # 4) ∑(combined_step_reward) 与 原始 ORM 符号一致的比例
    sum_step_reward_sign_agree = float(sum(sum_sign_agree_flags) / max(1, len(sum_sign_agree_flags)))  # ⭐ 计算综合奖励总和与原始 ORM 符号一致的比例

    # 5) PRM 标注与 ORM 的“全局一致性”（多数派）
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

    # 注意：返回 (rewards, stats) 二元组（仅 decouple 如此），其余方案仍然只返回 rewards
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
        s = torch.flip(torch.cumsum(torch.flip(t, dims=[0]), dim=0), dims=[0])  # ⭐ Compute the suffix sum by flipping, cumulatively summing, and then flipping back
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
        adv_i = torch.tensor(step_adv[i], device=device, dtype=torch.float32)  # ⭐ Convert step advantage values to a tensor
        sid_row = step_ids[i]
        valid = sid_row >= 0
        if torch.any(valid):
            sids = sid_row[valid]
            out[i, valid] = adv_i[sids]  # ⭐ Assign advantage values to the corresponding token positions
    return out

# =========================
# Entry
# =========================

def compute_prm_grpo_advantages(
    batch,                          # DataProto 或兼容结构：batch.batch[...] 可索引
    step_flags: List[List[bool]],   # 每条轨迹的 GOOD/BAD 标志
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

    # ---- 1. 数据准备阶段：提取必要字段 ----
    # 获取设备信息，确保所有张量在同一设备上
    responses = batch.batch["responses"]
    device = responses.device if torch.is_tensor(responses) else torch.as_tensor(responses).device

    # 提取step_ids和group_ids，并确保数据类型正确
    step_ids = _ensure_tensor(batch.batch["step_ids"], device=device, dtype=torch.long)      # (B, L_resp) with -1 for non-response
    # >>> add begin: 对齐到真实响应长度 <<<
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

    # ---- 2. 提取token-level奖励 ----
    # 尝试多种可能的字段名获取token-level奖励
    token_keys_try = ["token_level_rewards", "response_token_level_rewards", "token_rewards"]
    token_level_rewards = None
    for k in token_keys_try:
        if k in batch.batch:
            token_level_rewards = _ensure_tensor(batch.batch[k], device=device, dtype=torch.float32)
            break
    if token_level_rewards is None:
        raise KeyError("token-level rewards not found in batch (tried keys: token_level_rewards / response_token_level_rewards / token_rewards)")

    # ---- 3. ORM处理：计算ORM分数 ----
    # 对token-level奖励求和得到轨迹级ORM分数，用于各个方案的奖励构造
    orm_sum = token_level_rewards.sum(dim=1)   # (B,)
    # orm_scores = torch.where(orm_sum > 0, torch.ones_like(orm_sum), -torch.ones_like(orm_sum)).to(dtype=torch.float32)
    orm_scores = torch.where(orm_sum > 0.5, torch.ones_like(orm_sum), -torch.ones_like(orm_sum)).to(dtype=torch.float32)  # ⭐ Compute ORM scores based on token-level rewards

    # ---- 4. 方案选择阶段：根据scheme选择具体的奖励构造方案 ----
    extra_metrics = {}
    scheme = (scheme or "decouple").lower()

    if scheme == "allocation":
        # 方案2：allocation —— 一致性权重瓜分 + 组内减均值中心化
        step_rewards, extra_metrics = _build_allocation(orm_scores, step_flags, step_ids, group_ids, hyper)
    elif scheme == "decouple":
        # 方案4：decouple —— PRM和ORM分别标准化后组合
        step_rewards, extra_metrics = _build_decouple(orm_scores, step_flags, step_ids, group_ids, hyper,)
    else:
        raise ValueError(f"Unknown PRM scheme: {scheme} (expected one of: allocation | decouple)")

    # ---- 5. 优势值计算阶段：step后缀和 + 广播到token ----
    # 对step-level奖励进行后缀和计算得到step-level优势值
    step_adv = suffix_sum_on_steps(step_rewards)
    # 将step-level优势值广播到token-level
    advantages = broadcast_step_adv_to_tokens(step_adv, step_ids)

    # ---- 6. 结果返回阶段：构造返回字典 ----
    # 返回token-level优势值和原始ORM分数
    return {
        "advantages": advantages,        # (B, L_resp) token-level优势值
        "orm_scores": orm_scores,         # (B,) 逐条轨迹的 ±1
        "metrics":  extra_metrics,      # ✅ 仅 decouple 会有
    }
