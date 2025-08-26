# -*- coding: utf-8 -*-
# PRM step-reward → GRPO (group-normalize) → suffix-sum advantages
from __future__ import annotations
from typing import List, Dict, Tuple
import torch
from dataclasses import dataclass

@dataclass
class PRMHyper:
    alpha_pos: float = 1.0
    beta_pos:  float = 0.2
    alpha_neg: float = 1.0
    beta_neg:  float = 0.2
    eps:       float = 1e-8

def _step_end_indices_from_step_ids(step_ids_row: torch.Tensor) -> List[int]:
    ids = step_ids_row.tolist()
    end_ids = []
    last_sid, last_idx = None, None
    for t, sid in enumerate(ids):
        if sid < 0:
            continue
        if last_sid is None:
            last_sid, last_idx = sid, t
        elif sid != last_sid:
            end_ids.append(last_idx)
            last_sid, last_idx = sid, t
        else:
            last_idx = t
    if last_sid is not None:
        end_ids.append(last_idx)
    return end_ids

def compute_step_rewards_from_flags(
    orm: torch.Tensor,              # (B,)
    step_flags: List[List[bool]],   # GOOD/BAD flags
    step_ids: torch.Tensor,         # (B, L_resp)
    hyper: PRMHyper,
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    瓜分式构造 PRM：|O| 在 GOOD/BAD 步骤内均分。
    """
    B, _ = step_ids.shape
    step_rewards: List[List[float]] = []
    step_end_indices: List[List[int]] = []

    for i in range(B):
        ends = _step_end_indices_from_step_ids(step_ids[i])
        K = len(ends)
        flags = list(step_flags[i]) if i < len(step_flags) else []
        if K != len(flags):
            default_flag = True if orm[i].item() > 0 else False
            if len(flags) < K:
                flags = flags + [default_flag] * (K - len(flags))
            else:
                flags = flags[:K]

        O = orm[i].item()
        good_idx = [j for j, f in enumerate(flags) if f]
        bad_idx  = [j for j, f in enumerate(flags) if not f]
        n_g, n_b = max(1, len(good_idx)), max(1, len(bad_idx))

        if O >= 0:
            r_g =  hyper.alpha_pos * abs(O) / n_g
            r_b = -hyper.beta_pos  * abs(O) / n_b
        else:
            r_g =  hyper.beta_neg  * abs(O) / n_g
            r_b = -hyper.alpha_neg * abs(O) / n_b

        r = [0.0] * K
        for j in good_idx: r[j] = r_g
        for j in bad_idx:  r[j] = r_b

        step_rewards.append(r)
        step_end_indices.append(ends)

    return step_rewards, step_end_indices

def group_standardize_step_rewards(
    step_rewards: List[List[float]],
    group_ids: torch.Tensor,
    eps: float = 1e-8,
) -> List[List[float]]:
    gid_unique = group_ids.unique().tolist()
    gid_to_flat = {int(g): [] for g in gid_unique}
    for i, rewards in enumerate(step_rewards):
        gid_to_flat[int(group_ids[i].item())].extend(rewards)

    stats = {}
    for g, vals in gid_to_flat.items():
        if len(vals) == 0:
            stats[g] = (0.0, 1.0)
            continue
        t = torch.tensor(vals, dtype=torch.float32)
        mu = t.mean().item()
        sd = max(t.std(unbiased=False).item(), eps)
        stats[g] = (mu, sd)

    out: List[List[float]] = []
    for i, rewards in enumerate(step_rewards):
        mu, sd = stats[int(group_ids[i].item())]
        out.append([(v - mu) / sd for v in rewards])
    return out

def suffix_sum_advantages(
    step_rewards_norm: List[List[float]],
    step_end_indices: List[List[int]],
    resp_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    B = len(step_rewards_norm)
    adv = torch.zeros((B, resp_len), dtype=dtype, device=device)
    for i in range(B):
        for r, end_t in zip(step_rewards_norm[i], step_end_indices[i]):
            if 0 <= end_t < resp_len:
                adv[i, end_t] += float(r)
        if resp_len > 1:
            adv[i].flip(dims=[0]).cumsum_(dim=0).flip_(dims=[0])
    return adv

def compute_prm_grpo_advantages(
    batch,
    step_flags: List[List[bool]],
    group_size: int,
    hyper: PRMHyper = PRMHyper(),
) -> Dict[str, torch.Tensor]:
    responses = batch.batch["responses"]
    resp_len  = responses.size(1)
    device    = responses.device

    step_ids = batch.batch["step_ids"][:, -resp_len:]
    token_level_rewards = batch.batch["token_level_rewards"]

    # 直接用 token_level_rewards 作为 ORM（和 Ray GRPO 一致）
    orm = token_level_rewards.sum(dim=1)  # (B,)

    raw_step_rewards, step_end_indices = compute_step_rewards_from_flags(
        orm=orm, step_flags=step_flags, step_ids=step_ids, hyper=hyper
    )

    # 直接使用 env_manager 注入的 group_ids，避免串组
    group_ids = batch.batch["group_ids"]
    if not torch.is_tensor(group_ids):
        group_ids = torch.as_tensor(group_ids)
    group_ids = group_ids.to(device=device, dtype=torch.long).view(-1)
    assert group_ids.numel() == responses.size(0), "group_ids length must equal batch size"

    norm_step_rewards = group_standardize_step_rewards(raw_step_rewards, group_ids, eps=hyper.eps)

    A = suffix_sum_advantages(norm_step_rewards, step_end_indices, resp_len, device=device)

    return {
        "advantages": A,
        "orm_scalar": orm,
    }
