import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss



def het_compute_token_on_off_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    off_cliprange_high=1.0,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Computes the on-policy and off-policy losses for reinforcement learning using PPO.

    Args:
        old_log_prob (Tensor): Log probabilities of the actions under the old policy.
        log_prob (Tensor): Log probabilities of the actions under the new policy.
        advantages (Tensor): Advantage values for the actions.
        response_mask (Tensor): Mask indicating which tokens are part of the response.
        exp_mask (Tensor): Mask indicating which tokens are part of the experience replay.
        cliprange (float, optional): Clipping range for the policy ratio.
        cliprange_low (float, optional): Lower bound for the clipping range.
        cliprange_high (float, optional): Upper bound for the clipping range.
        off_cliprange_high (float, optional): Upper bound for the off-policy clipping range.
        clip_ratio_c (float, optional): Constant used in the clipping mechanism.
        loss_agg_mode (str, optional): Mode for aggregating the losses. Defaults to "token-mean".

    Returns:
        dict: A dictionary containing various computed losses and metrics.
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    ratio = torch.exp(negative_approx_kl)

    def compute_pg_losses(cliprange_low, cliprange_high):
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
        clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)
        return pg_losses, clipfrac, clipfrac_lower

    # On-policy calculations
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_pg_losses, on_pg_clipfrac, on_pg_clipfrac_lower = compute_pg_losses(cliprange_low, cliprange_high)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)  # ⭐ Compute the on-policy loss

    # Off-policy calculations
    off_cliprange_low = cliprange_low
    off_pg_losses, off_pg_clipfrac, off_pg_clipfrac_lower = compute_pg_losses(off_cliprange_low, off_cliprange_high)
    off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)  # ⭐ Compute the off-policy loss
    off_pg_loss = torch.tensor(0.0) if off_pg_loss.isnan().item() else off_pg_loss

    # Combine on-policy and off-policy losses
    exp_mask = exp_mask.float()
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)  # ⭐ Aggregate the combined losses

    return {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses": on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
    }



def bam_compute_token_on_off_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,   # (bs, response_length) ANNI add: 1 indicates off-policy data; 0 indicates on-policy data
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Computes the on-policy and off-policy losses for reinforcement learning.

    Args:
        old_log_prob (Tensor): Log probabilities of the old policy.
        log_prob (Tensor): Log probabilities of the current policy.
        advantages (Tensor): Advantage values.
        response_mask (Tensor): Mask indicating valid response tokens.
        exp_mask (Tensor): Mask indicating whether the data is from an off-policy (1) or on-policy (0) source.
        cliprange (float, optional): Clipping range for PPO. Defaults to None.
        cliprange_low (float, optional): Lower clipping range for PPO. Defaults to None.
        cliprange_high (float, optional): Upper clipping range for PPO. Defaults to None.
        clip_ratio_c (float, optional): Clipping ratio constant. Defaults to 3.0.
        loss_agg_mode (str, optional): Mode for aggregating the loss. Defaults to "token-mean".

    Returns:
        dict: A dictionary containing various computed losses and metrics.
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # on-policy: no changes
    # off-policy: denominator=1 + reshape + no clipping

    # on-policy: keep unchanged
    ratio = torch.exp(negative_approx_kl)   # (bs, response_length)
    on_pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    on_clip_pg_losses1 = torch.maximum(on_pg_losses1, on_pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses1).float(), response_mask)

    on_pg_losses3 = -advantages * clip_ratio_c
    on_clip_pg_losses2 = torch.min(on_pg_losses3, on_clip_pg_losses1)
    on_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(on_clip_pg_losses1, on_pg_losses3) * (advantages < 0).float(), response_mask)

    on_pg_losses = torch.where(advantages < 0, on_clip_pg_losses2, on_clip_pg_losses1)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0-exp_mask) * response_mask)

    # off-policy
    off_ratio = torch.exp(log_prob)     #(bs, response_length)
    off_ratio = off_ratio / (off_ratio + 0.1)   # ⭐ Reshape the off-policy ratio to stabilize the loss
    off_pg_losses = -advantages * off_ratio
    off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)

    exp_mask = exp_mask.float()
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    ret_dict = {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses":  on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
    }

    return ret_dict

def bam_compute_token_on_off_policy_loss_v2(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,   # (bs, response_length) ANNI add: 1 indicates off-policy data; 0 indicates on-policy data
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Computes the on-policy and off-policy losses for reinforcement learning, using various methods to handle clipping and masking.

    Args:
        old_log_prob (Tensor): The old log probabilities of the actions.
        log_prob (Tensor): The new log probabilities of the actions.
        advantages (Tensor): The advantage values.
        response_mask (Tensor): A mask indicating which tokens are part of the response.
        exp_mask (Tensor): A mask indicating whether the data is from an off-policy (1) or on-policy (0) source.
        cliprange (float, optional): The range for clipping the ratio. Defaults to None.
        cliprange_low (float, optional): The lower bound for clipping the ratio. Defaults to None.
        cliprange_high (float, optional): The upper bound for clipping the ratio. Defaults to None.
        clip_ratio_c (float, optional): The clipping ratio constant. Defaults to 3.0.
        loss_agg_mode (str, optional): The mode for aggregating the loss. Defaults to "token-mean".

    Returns:
        dict: A dictionary containing the computed losses and other relevant metrics.
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # on-policy: no changes
    # off-policy: denominator=1 + reshape + no clipping

    # on-policy: keep unchanged
    ratio = torch.exp(negative_approx_kl)   # (bs, response_length)
    on_pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    on_clip_pg_losses1 = torch.maximum(on_pg_losses1, on_pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses1).float(), response_mask)

    on_pg_losses3 = -advantages * clip_ratio_c
    on_clip_pg_losses2 = torch.min(on_pg_losses3, on_clip_pg_losses1)
    on_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(on_clip_pg_losses1, on_pg_losses3) * (advantages < 0).float(), response_mask)

    on_pg_losses = torch.where(advantages < 0, on_clip_pg_losses2, on_clip_pg_losses1)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0-exp_mask) * response_mask)

    # off-policy
    off_ratio = torch.exp(log_prob)     #(bs, response_length)
    off_ratio = off_ratio / (off_ratio + 0.1)   # ⭐ Reshape the off-policy ratio
    off_pg_losses = -advantages * off_ratio
    ############
    # ANNI add 0728: For negative samples with A<0, do not compute loss gradients, mask them out
    off_positive_mask = (exp_mask > 0) & (advantages >=0) & (response_mask > 0) # mask containing only off-policy data with advantages>=0
    adjusted_off_pg_losses = torch.where(off_positive_mask, off_pg_losses, torch.zeros_like(off_pg_losses))
    off_pg_loss = verl_F.masked_mean(off_pg_losses, off_positive_mask)
    if torch.isnan(off_pg_loss).item():
        off_pg_loss = torch.tensor(0.0)

    exp_mask = exp_mask.float()
    pg_losses = adjusted_off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    ret_dict = {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses":  on_pg_losses,
        "off_pg_losses": off_pg_losses,
        # "adjusted_off_pg_losses": adjusted_off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
    }

    return ret_dict

def bam_compute_token_on_off_policy_loss_v3(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,   # (bs, response_length) ANNI add: 1 indicates off-policy data; 0 indicates on-policy data
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Computes the on-policy and off-policy losses for reinforcement learning, using various methods to handle clipping and masking.

    Args:
        old_log_prob (Tensor): The log probability of the old policy.
        log_prob (Tensor): The log probability of the new policy.
        advantages (Tensor): The advantage values.
        response_mask (Tensor): A mask indicating which tokens are part of the response.
        exp_mask (Tensor): A mask indicating whether the data is from an off-policy (1) or on-policy (0) source.
        cliprange (float, optional): The range for clipping the ratio.
        cliprange_low (float, optional): The lower bound for clipping the ratio.
        cliprange_high (float, optional): The upper bound for clipping the ratio.
        clip_ratio_c (float, optional): The constant used for clipping the ratio.
        loss_agg_mode (str, optional): The mode for aggregating the loss. Default is "token-mean".

    Returns:
        dict: A dictionary containing the computed losses and other relevant metrics.
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # on-policy: no changes
    # off-policy: cliphigh=1, rest kept consistent with on-policy

    # on-policy: keep unchanged
    ratio = torch.exp(negative_approx_kl)   # (bs, response_length) ⭐ Compute the ratio of new to old policy probabilities
    on_pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    on_clip_pg_losses1 = torch.maximum(on_pg_losses1, on_pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses1).float(), response_mask)

    on_pg_losses3 = -advantages * clip_ratio_c
    on_clip_pg_losses2 = torch.min(on_pg_losses3, on_clip_pg_losses1)
    on_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(on_clip_pg_losses1, on_pg_losses3) * (advantages < 0).float(), response_mask)

    on_pg_losses = torch.where(advantages < 0, on_clip_pg_losses2, on_clip_pg_losses1)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0-exp_mask) * response_mask)

    # off-policy
    off_pg_losses1 = -advantages * ratio
    off_cliprange_low = cliprange_low
    off_cliprange_high = 1.0
    off_pg_losses2 = -advantages * torch.clamp(ratio, 1 - off_cliprange_low, 1 + off_cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    off_clip_pg_losses1 = torch.maximum(off_pg_losses1, off_pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    off_pg_clipfrac = verl_F.masked_mean(torch.gt(off_pg_losses2, off_pg_losses1).float(), response_mask)
    off_pg_losses3 = -advantages * clip_ratio_c
    off_clip_pg_losses2 = torch.min(off_pg_losses3, off_clip_pg_losses1)
    off_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(off_clip_pg_losses1, off_pg_losses3) * (advantages < 0).float(), response_mask)

    off_pg_losses = torch.where(advantages < 0, off_clip_pg_losses2, off_clip_pg_losses1)
    off_pg_loss = verl_F.masked_mean(off_pg_losses, (1.0-exp_mask) * response_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)

    exp_mask = exp_mask.float()
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    ret_dict = {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses":  on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
    }

    return ret_dict