# shuchang: 0809
# FIXME: 这个文件是step_parser.py，功能：把解析模型的response_id解析为step，统一所有需要step的模块
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch

@dataclass
class StepParseResult:
    segments: List[Dict]         # [{'role': str, 'start': int, 'end': int, 'tokens': List[int]}]
    steps: List[Dict]            # [{'action_tokens': List[int], 'observation_tokens': List[int],
                                #   'action_text': str, 'observation_text': str, 
                                #   'action_start': int, 'action_end': int, 'obs_start': int, 'obs_end': int}]
    step_ids: List[int]          # len == len(response_ids); assistant动作区间标k，其余-1

def _find_first_subseq(hay, needle):
    """安全的子序列搜索，避免单token误匹配"""
    if not needle:
        return None
    L = len(needle)
    for i in range(len(hay) - L + 1):
        if hay[i:i+L] == needle:
            return i
    return None

def _locate_template_positions(tokens: List[int], tpl: List[int]) -> List[int]:
    """返回 tpl 在 tokens 中出现的起点索引"""
    if not tpl:  # 保护：避免空模板死循环
        return []
    
    pos, out, L = 0, [], len(tpl)
    while pos <= len(tokens) - L:
        if tokens[pos:pos+L] == tpl:
            out.append(pos)
            pos += L
        else:
            pos += 1
    return out

def _extract_role_header_tokens(tokenizer, role: str) -> List[int]:
    """
    通用方法：自动提取任何模型的role header tokens
    原理：通过对比空内容和带内容的消息，找出role header部分
    如果提取失败，直接抛出异常
    """
    try:
        if role == "assistant":
            # 比较不带assistant回复 vs 带assistant回复的差异
            user_only = [{"role": "user", "content": ""}]
            user_tokens = tokenizer.apply_chat_template(
                user_only, tokenize=True, add_generation_prompt=False
            )
            
            # 带assistant的完整对话
            full_dialog = [{"role": "user", "content": ""}, {"role": "assistant", "content": "x"}]
            full_tokens = tokenizer.apply_chat_template(
                full_dialog, tokenize=True, add_generation_prompt=False
            )
            
            # 找到"x"的位置（使用安全的子序列搜索）
            x_tokens = tokenizer.encode("x", add_special_tokens=False)
            if not x_tokens:
                raise ValueError(f"Cannot encode 'x' token for role {role}")
            
            
            x_position = _find_first_subseq(full_tokens, x_tokens)
            if x_position is None:
                raise ValueError(f"Cannot find 'x' token sequence in full dialog for role {role}")
            
            
            # assistant header = 从user_only结束到"x"开始的部分
            user_len = len(user_tokens)
            
            if user_len < x_position:
                header_tokens = full_tokens[user_len:x_position]
                return header_tokens
            elif user_len == x_position:
                return []  # 返回空header，这是合法情况
            else:
                raise ValueError(f"Invalid token positions for role {role}: user_len={user_len}, x_pos={x_position}")
                
        else:
            # 对于user等其他角色：比较空内容vs带内容
            # 关键修复：不要让user模板包含system message
            empty_msg = [{"role": role, "content": ""}]
            empty_tokens = tokenizer.apply_chat_template(
                empty_msg, tokenize=True, add_generation_prompt=False
            )
            
            content_msg = [{"role": role, "content": "x"}]
            content_tokens = tokenizer.apply_chat_template(
                content_msg, tokenize=True, add_generation_prompt=False
            )
            
            # 找到"x"的位置（使用安全的子序列搜索）
            x_tokens = tokenizer.encode("x", add_special_tokens=False)
            if not x_tokens:
                raise ValueError(f"Cannot encode 'x' token for role {role}")
            
            x_position = _find_first_subseq(content_tokens, x_tokens)
            if x_position is None:
                raise ValueError(f"Cannot find 'x' token sequence in content message for role {role}")
            
            # 关键修复：用更精确的方法提取纯role header
            # 对于content_msg，x前面的部分应该是role header
            # 但如果empty_tokens包含了额外内容（如system），需要排除
            
            if len(content_tokens) > len(empty_tokens):
                # 新增部分是 header + "x"
                added_part = content_tokens[len(empty_tokens):]
                x_pos_in_added = _find_first_subseq(added_part, x_tokens)
                if x_pos_in_added is not None:
                    header_tokens = added_part[:x_pos_in_added]
                else:
                    # fallback: 直接取x之前的部分
                    header_tokens = content_tokens[:x_position]
            else:
                # 直接从开始到x位置
                header_tokens = content_tokens[:x_position]
            
            # 额外验证：如果header太长（包含system message），尝试提取纯role部分
            header_decoded = tokenizer.decode(header_tokens)
            
            # 如果包含system message，尝试只取最后的role部分
            if f"<|im_start|>{role}" in header_decoded:
                # 找到最后一个role标记的位置
                role_marker = f"<|im_start|>{role}\n"
                role_tokens = tokenizer.encode(role_marker, add_special_tokens=False)
                
                # 在header_tokens中找到role_tokens的位置
                role_pos = _find_first_subseq(header_tokens, role_tokens)
                if role_pos is not None:
                    # 只取role标记部分
                    header_tokens = role_tokens
            return header_tokens
            
    except Exception as e:
        # 不要fallback，直接报错
        raise RuntimeError(f"Failed to extract header tokens for role '{role}': {e}") from e

def parse_response_ids_to_steps(
    response_ids: List[int],
    tokenizer,
    assistant_tpl: List[int] = None,
    user_tpl: List[int] = None,
    mark_observation: bool = False,  # 是否给observation也打step_id
) -> StepParseResult:
    """
    改进版实现：
    1. 按tokenid切分，保留start/end索引
    2. 合并相邻same role，保证交替且第一个是assistant  
    3. 组对成steps：每个assistant + 可选的后续user
    4. 在原长度数组上**原位标记**step_ids，避免错位风险
    """
    
    # 自动获取模板（通用于所有模型）
    if assistant_tpl is None:
        assistant_tpl = _extract_role_header_tokens(tokenizer, "assistant")
    if user_tpl is None:
        user_tpl = _extract_role_header_tokens(tokenizer, "user")

    # Step 1: 按照tokenid切分，保留start/end索引
    a_template_starts = _locate_template_positions(response_ids, assistant_tpl)
    u_template_starts = _locate_template_positions(response_ids, user_tpl)
    
    # 计算body开始位置（处理空模板的情况）
    if assistant_tpl:
        a_body_starts = [pos + len(assistant_tpl) for pos in a_template_starts]
    else:
        # 没有assistant header：从0开始，表示response直接从assistant内容开始
        a_body_starts = [0] if response_ids else []
        
    if user_tpl:
        u_body_starts = [pos + len(user_tpl) for pos in u_template_starts]
    else:
        u_body_starts = []

    # 创建所有切分点，保持原始位置信息
    if assistant_tpl:
        # 正常情况：有assistant header
        markers: List[Tuple[int, str]] = (
            [(pos, "assistant") for pos in a_body_starts] +
            [(pos, "user") for pos in u_body_starts] +
            [(len(response_ids), "end")]
        )
    else:
        # 特殊情况：没有assistant header，从0开始，全靠user切分
        markers: List[Tuple[int, str]] = (
            [(0, "assistant")] +  # 从开头开始就是assistant
            [(pos, "user") for pos in u_body_starts] +
            [(len(response_ids), "end")]
        )
    markers.sort(key=lambda x: x[0])

    # 构造segments，保留start/end索引
    raw_segments = []
    for i, (start, role) in enumerate(markers[:-1]):
        end = markers[i + 1][0]
        if start < end and role != "end":
            segment_tokens = response_ids[start:end]
            raw_segments.append({
                "role": role, 
                "start": start, 
                "end": end,
                "tokens": segment_tokens
            })

    if not raw_segments:
        return StepParseResult(
            segments=[], 
            steps=[], 
            step_ids=[-1] * len(response_ids)
        )

    # Step 2: 合并相邻same role，更新start/end边界
    merged_segments = []
    for seg in raw_segments:
        if (merged_segments and 
            merged_segments[-1]["role"] == seg["role"] and
            merged_segments[-1]["end"] == seg["start"]):
            # 合并相邻同role：扩展end边界，合并tokens
            merged_segments[-1]["end"] = seg["end"]
            merged_segments[-1]["tokens"].extend(seg["tokens"])
        else:
            merged_segments.append({
                "role": seg["role"],
                "start": seg["start"],
                "end": seg["end"],
                "tokens": seg["tokens"].copy()
            })

    # 确保第一个是assistant
    while merged_segments and merged_segments[0]["role"] != "assistant":
        merged_segments.pop(0)

    if not merged_segments:
        return StepParseResult(
            segments=[], 
            steps=[], 
            step_ids=[-1] * len(response_ids)
        )

    # Step 3: 组对成steps，保留位置信息（聚合到下一个assistant之前）
    steps = []
    i = 0
    while i < len(merged_segments):
        seg = merged_segments[i]
        if seg["role"] != "assistant":
            i += 1
            continue
            
        # action = 当前assistant段
        action_start, action_end = seg["start"], seg["end"]
        action_tokens = seg["tokens"]
        action_text = tokenizer.decode(action_tokens, skip_special_tokens=True)
        
        # 从 i+1 开始，直到遇到下一个 assistant 为止，全部并成 observation
        j = i + 1
        obs_start = action_end
        obs_end = obs_start
        obs_tokens = []
        
        while j < len(merged_segments) and merged_segments[j]["role"] != "assistant":
            obs_end = merged_segments[j]["end"]
            obs_tokens.extend(merged_segments[j]["tokens"])
            j += 1
            
        obs_text = tokenizer.decode(obs_tokens, skip_special_tokens=True) if obs_tokens else ""
        
        step = {
            "action_tokens": action_tokens,
            "observation_tokens": obs_tokens,
            "action_text": action_text,
            "observation_text": obs_text,
            "action_start": action_start,
            "action_end": action_end,
            "obs_start": obs_start,
            "obs_end": obs_end,
        }
        steps.append(step)
        
        i = j  # 跳到下一个 assistant（或结束）

    # Step 4: 在原长度数组上**原位标记**step_ids
    step_ids = [-1] * len(response_ids)
    
    for step_k, step in enumerate(steps):
        # 标记assistant动作区间
        for pos in range(step["action_start"], step["action_end"]):
            step_ids[pos] = step_k
            
        # 可选：标记observation区间
        if mark_observation and step["obs_start"] < step["obs_end"]:
            for pos in range(step["obs_start"], step["obs_end"]):
                step_ids[pos] = step_k

    return StepParseResult(
        segments=merged_segments,
        steps=steps,
        step_ids=step_ids
    )

# 添加验证函数
def verify_step_alignment(batch, tokenizer, global_step):
    """验证语义评估和advantage scaling的step对齐"""
    print(f"\n=== Step Alignment Check (Step {global_step}) ===")
    
    batch_size = len(batch.batch["prompts"])
    alignment_errors = 0
    
    for sample_idx in range(min(5, batch_size)):  # 检查前5个样本
        # 从语义评估获取的steps
        semantic_steps = batch.non_tensor_batch["steps"][sample_idx]
        
        # 从step_ids获取的step数量
        step_ids = batch.batch["step_ids"][sample_idx]
        max_step_id = int(step_ids.max().item()) if (step_ids >= 0).any() else -1
        advantage_steps = max_step_id + 1 if max_step_id >= 0 else 0
        
        # 检查对齐
        semantic_count = len(semantic_steps)
        if semantic_count != advantage_steps:
            print(f"❌ Sample {sample_idx}: semantic={semantic_count}, advantage={advantage_steps}")
            alignment_errors += 1
        else:
            print(f"✅ Sample {sample_idx}: {semantic_count} steps aligned")
    
    if alignment_errors == 0:
        print("✅ [Alignment Great] All checked samples have aligned step counts!")
        return True
    else:
        print(f"❌ [Alignment Error] Found {alignment_errors} alignment errors!")
        return False
    
def verify_step_content(batch, tokenizer, sample_idx=0):
    """验证step内容的一致性"""
    print(f"\n=== Step Content Check (Sample {sample_idx}) ===")
    
    # 从batch获取
    response_tokens = batch.batch["responses"][sample_idx].tolist()
    step_ids = batch.batch["step_ids"][sample_idx].tolist()
    semantic_steps = batch.non_tensor_batch["steps"][sample_idx]
    
    # 重新解析验证
    from beyondagent.utils.step_parser import parse_response_ids_to_steps
    parse_result = parse_response_ids_to_steps(response_tokens, tokenizer)
    
    print(f"Parsed {len(parse_result.steps)} steps:")
    for i, step in enumerate(parse_result.steps):
        semantic_step = semantic_steps[i] if i < len(semantic_steps) else {"action": "MISSING", "observation": "MISSING"}
        print(f"Step {i}:")
        print(f"  Parsed Action: {step['action_text'][:50]}...")
        print(f"  Semantic Action: {semantic_step.get('action', 'MISSING')[:50]}...")
        print(f"  Match: {step['action_text'].strip() == semantic_step.get('action', '').strip()}")

# 测试函数
def test_parse_response_ids_to_steps():
    """测试函数，验证通用模板提取和原位标记的正确性"""
    from transformers import AutoTokenizer
    
        # 测试多种模型
    test_models = [
        "/mnt/data/zouanni.zan/models/Qwen2.5-14B-Instruct",
        # "/mnt/data/taoshuchang.tsc/models/gemma-3-12b-it/",  # 如果有其他模型可以测试
        # "meta-llama/Llama-2-7b-chat-hf"
    ]
    
    for model_name in test_models:
        print(f"\n{'='*50}")
        print(f"测试模型: {model_name}")
        print(f"{'='*50}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 测试模板提取
            print("\n=== 自动提取的模板 ===")
            assistant_tpl = _extract_role_header_tokens(tokenizer, "assistant")
            user_tpl = _extract_role_header_tokens(tokenizer, "user")
            
            print(f"Assistant模板: {assistant_tpl}")
            print(f"Assistant解码: '{tokenizer.decode(assistant_tpl)}'")
            print(f"User模板: {user_tpl}")
            print(f"User解码: '{tokenizer.decode(user_tpl)}'")
            
            # 构造测试用例
            test_messages = [
                {"role": "user", "content": "帮我计算1+1"},
                {"role": "assistant", "content": "我来计算1+1。"},
                {"role": "user", "content": "结果assistant是多少？"},
                {"role": "assistant", "content": "结果是2。"},
            ]
            
            # 重要：应该只包含response部分（从第一个assistant开始）
            response_messages = test_messages[1:]  # 从assistant开始的所有消息
            response_text = tokenizer.apply_chat_template(response_messages, tokenize=False, add_generation_prompt=False)
            response_tokens = tokenizer.encode(response_text, add_special_tokens=False)

            prompt_ids = tokenizer.apply_chat_template(test_messages[:1], tokenize=True, add_generation_prompt=True)
            full_ids = tokenizer.apply_chat_template(test_messages, tokenize=True)
            response_tokens = full_ids[len(prompt_ids):]
            print(f"\n=== decoded response_tokens ===")
            print(tokenizer.decode(response_tokens))

            
            # 测试解析
            print(f"\n=== 测试Step解析 ===")
            result = parse_response_ids_to_steps(response_tokens, tokenizer, mark_observation=False)
            
            print(f"Response tokens总长度: {len(response_tokens)}")
            print(f"Step IDs长度: {len(result.step_ids)}")
            print(f"解析到 {len(result.steps)} 个steps:")
            
            for i, step in enumerate(result.steps):
                print(f"\nStep {i}:")
                print(f"  Action ({step['action_start']}-{step['action_end']}): {step['action_text'][:50]}")
                print(f"  Observation ({step['obs_start']}-{step['obs_end']}): {step['observation_text'][:50]}")
            
            # 验证标记正确性
            print(f"\n=== 验证标记正确性 ===")
            valid = True
            for pos, step_id in enumerate(result.step_ids):
                if step_id >= 0:
                    step = result.steps[step_id]
                    if not (step['action_start'] <= pos < step['action_end']):
                        print(f"❌ 错位检测: 位置{pos}标记为step{step_id}，但不在其action范围内")
                        valid = False
                        break
            
            
        except Exception as e:
            print(f"❌ 模型{model_name}测试失败: {e}")
            continue
    
    return True

if __name__ == "__main__":
    test_parse_response_ids_to_steps()