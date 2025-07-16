import torch
import numpy as np

def debug_advantage_structure(batch, tokenizer=None, sample_idx=None):
    """
    调试函数：检查advantage的结构是否符合GRPO预期
    
    Args:
        batch: DataProto batch
        tokenizer: tokenizer for decoding
        sample_idx: 指定要分析的样本索引，如果为None则自动找第一个非零advantage样本
    """
    print("="*50)
    print("ADVANTAGE STRUCTURE DEBUG")
    print("="*50)
    
    advantages = batch.batch["advantages"]
    response_mask = batch.batch.get("response_mask", None)
    
    # 获取loss_mask的response部分
    loss_mask = None
    if "loss_mask" in batch.batch:
        response_length = advantages.shape[1]  # response_len
        loss_mask = batch.batch["loss_mask"][:, -response_length:]  # 取response部分
        print(f"Using loss_mask for analysis (multi-turn mode)")
    else:
        print(f"Using response_mask for analysis (single-turn mode)")
    
    print(f"Advantages shape: {advantages.shape}")
    print(f"Advantages dtype: {advantages.dtype}")
    print(f"Batch size: {advantages.shape[0]}")
    
    # 自动找到第一个有非零advantage的样本
    if sample_idx is None:
        from beyondagent.module.advantage_assignment.parallel_semantic_assignment import _get_overall_advantage
        for i in range(len(advantages)):
            effective_mask = loss_mask[i] if loss_mask is not None else (response_mask[i] if response_mask is not None else None)
            adv_val = _get_overall_advantage(advantages[i], effective_mask)
            if abs(adv_val) > 1e-8:
                sample_idx = i
                print(f"🎯 Auto-selected sample {sample_idx} (first non-zero advantage: {adv_val:.6f})")
                break
        
        if sample_idx is None:
            sample_idx = 0
            print(f"⚠️  No non-zero advantage found, using sample 0")
    
    # 检查选定样本的详细信息
    sample_adv = advantages[sample_idx]
    print(f"\nSample {sample_idx} analysis:")
    print(f"  Shape: {sample_adv.shape}")
    print(f"  Min: {sample_adv.min().item():.6f}")
    print(f"  Max: {sample_adv.max().item():.6f}")
    print(f"  Mean: {sample_adv.mean().item():.6f}")
    print(f"  Std: {sample_adv.std().item():.6f}")
    
    # 检查非零值的情况
    non_zero_mask = torch.abs(sample_adv) > 1e-8
    non_zero_values = sample_adv[non_zero_mask]
    
    print(f"  Non-zero count: {non_zero_mask.sum().item()}")
    if len(non_zero_values) > 0:
        print(f"  Non-zero values unique count: {torch.unique(non_zero_values).shape[0]}")
        print(f"  First non-zero value: {non_zero_values[0].item():.6f}")
        if len(non_zero_values) > 1:
            print(f"  All non-zero values same?: {torch.allclose(non_zero_values, non_zero_values[0])}")
    
    # 分析有效token的advantage（优先使用loss_mask）
    effective_mask = loss_mask[sample_idx] if loss_mask is not None else (response_mask[sample_idx] if response_mask is not None else None)
    
    if effective_mask is not None:
        valid_advantages = sample_adv[effective_mask.bool()]
        mask_name = "loss_mask" if loss_mask is not None else "response_mask"
        print(f"  Valid tokens count ({mask_name}): {effective_mask.sum().item()}")
        if len(valid_advantages) > 0:
            print(f"  Valid advantages unique count: {torch.unique(valid_advantages).shape[0]}")
            print(f"  All valid advantages same?: {torch.allclose(valid_advantages, valid_advantages[0]) if len(valid_advantages) > 1 else True}")
    
    # 检查使用不同方法计算的overall advantage
    print(f"\nOverall advantage calculation methods:")
    
    # 方法1：错误的sum方法
    wrong_sum = sample_adv.sum().item()
    print(f"  Wrong (sum): {wrong_sum:.6f}")
    
    # 方法2：正确的方法（使用loss_mask）
    from beyondagent.module.advantage_assignment.parallel_semantic_assignment import _get_overall_advantage
    correct_value = _get_overall_advantage(sample_adv, effective_mask)
    print(f"  Correct (using proper mask): {correct_value:.6f}")
    
    # 方法3：仅取第一个有效值
    if effective_mask is not None:
        valid_advantages = sample_adv[effective_mask.bool()]
        if len(valid_advantages) > 0:
            first_valid = valid_advantages[0].item()
            print(f"  First valid token: {first_valid:.6f}")
    
    # 检查是否所有样本都遵循GRPO模式
    print(f"\nBatch-level GRPO pattern check:")
    grpo_compliant_samples = 0
    for i in range(min(5, advantages.shape[0])):  # 只检查前5个样本
        sample = advantages[i]
        
        # 使用正确的mask
        if loss_mask is not None:
            valid_vals = sample[loss_mask[i].bool()]
        elif response_mask is not None:
            valid_vals = sample[response_mask[i].bool()]
        else:
            valid_vals = sample[torch.abs(sample) > 1e-8]
        
        if len(valid_vals) > 1:
            all_same = torch.allclose(valid_vals, valid_vals[0], atol=1e-6)
            print(f"  Sample {i}: {len(valid_vals)} valid tokens, all same: {all_same}")
            if all_same:
                grpo_compliant_samples += 1
        else:
            print(f"  Sample {i}: {len(valid_vals)} valid tokens")
            grpo_compliant_samples += 1
    
    print(f"  GRPO compliant: {grpo_compliant_samples}/{min(5, advantages.shape[0])}")
    
    # 如果有step_ids，检查step级别的advantage分布
    if "step_ids" in batch.batch:
        step_ids = batch.batch["step_ids"][sample_idx]
        print(f"\nStep-level advantage analysis for sample {sample_idx}:")
        unique_step_ids = torch.unique(step_ids[step_ids >= 0])
        for step_id in unique_step_ids:
            step_mask = (step_ids == step_id)
            step_advantages = sample_adv[step_mask]
            if len(step_advantages) > 0:
                print(f"  Step {step_id.item()}: {len(step_advantages)} tokens, "
                      f"value: {step_advantages[0].item():.6f}, "
                      f"all same: {torch.allclose(step_advantages, step_advantages[0]) if len(step_advantages) > 1 else True}")
    
    # 新增：loss_mask与step_ids的对应关系检查
    if "step_ids" in batch.batch and loss_mask is not None:
        print(f"\nStep-ids vs Loss-mask correspondence check:")
        step_ids = batch.batch["step_ids"][sample_idx]
        sample_loss_mask = loss_mask[sample_idx]
        
        for step_id in torch.unique(step_ids[step_ids >= 0]):
            step_mask = (step_ids == step_id)
            step_loss_mask_values = sample_loss_mask[step_mask]
            
            if len(step_loss_mask_values) > 0:
                all_trainable = step_loss_mask_values.all().item()
                any_trainable = step_loss_mask_values.any().item()
                print(f"  Step {step_id.item()}: {step_mask.sum().item()} tokens, "
                      f"all trainable: {all_trainable}, any trainable: {any_trainable}")
    
    print("="*50)

def validate_grpo_advantage_structure(advantages, response_mask=None, loss_mask=None, tolerance=1e-6):
    """
    验证advantage结构是否符合GRPO要求
    
    Args:
        advantages: advantage tensor, shape (bs, resp_len)
        response_mask: response mask (deprecated, use loss_mask instead)
        loss_mask: loss mask for response part, shape (bs, resp_len)
        tolerance: tolerance for floating point comparison
    
    Returns:
        bool: 是否符合GRPO结构
        str: 详细说明
    """
    issues = []
    
    batch_size, seq_len = advantages.shape
    
    # 优先使用loss_mask
    effective_mask = loss_mask if loss_mask is not None else response_mask
    
    for i in range(batch_size):
        sample_adv = advantages[i]
        
        if effective_mask is not None:
            valid_advantages = sample_adv[effective_mask[i].bool()]
        else:
            valid_advantages = sample_adv[torch.abs(sample_adv) > 1e-8]
        
        if len(valid_advantages) == 0:
            continue  # 跳过全零的样本
        
        # 检查所有有效token是否有相同的advantage
        if len(valid_advantages) > 1:
            if not torch.allclose(valid_advantages, valid_advantages[0], atol=tolerance):
                unique_vals = torch.unique(valid_advantages)
                issues.append(f"Sample {i}: Found {len(unique_vals)} different advantage values: {unique_vals[:5].tolist()}")
    
    is_valid = len(issues) == 0
    mask_type = "loss_mask" if loss_mask is not None else ("response_mask" if response_mask is not None else "non-zero values")
    
    if is_valid:
        return True, f"✅ All samples follow GRPO advantage structure (same value for all valid tokens using {mask_type})"
    else:
        return False, f"❌ GRPO structure violations (using {mask_type}):\n" + "\n".join(issues[:5])

def add_debug_to_trainer_fit():
    """
    在fit函数中添加调试代码的建议位置
    """
    debug_code = '''
    # 在compute_advantage之后，semantic处理之前添加：
    print("🔍 [DEBUG] Checking advantage structure before semantic processing...")
    debug_advantage_structure(batch, self.tokenizer)  # 自动选择非零样本
    
    # 检查前几个样本的advantage值范围
    advs = batch.batch["advantages"]
    print(f"🔍 [DEBUG] Advantage stats - Shape: {advs.shape}")
    print(f"🔍 [DEBUG] Advantage range: [{advs.min().item():.6f}, {advs.max().item():.6f}]")
    print(f"🔍 [DEBUG] Advantage mean: {advs.mean().item():.6f}")
    
    # 使用正确的loss_mask进行验证
    response_length = advs.shape[1]
    loss_mask_response = batch.batch["loss_mask"][:, -response_length:]
    
    is_valid, message = validate_grpo_advantage_structure(
        advs, 
        loss_mask=loss_mask_response
    )
    print(f"🔍 [GRPO Validation] {message}")
    
    # 检查是否有advantage=0的样本（使用正确的mask）
    from beyondagent.module.advantage_assignment.parallel_semantic_assignment import _get_overall_advantage
    zero_count = 0
    for i in range(len(batch)):
        adv_val = _get_overall_advantage(advs[i], loss_mask_response[i])
        if abs(adv_val) < 1e-8:
            zero_count += 1
    
    print(f"🔍 [Zero Advantage] {zero_count}/{len(batch)} samples have advantage≈0 (using loss_mask)")
    '''
    return debug_code