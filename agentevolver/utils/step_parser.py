# shuchang: 0809
# FIXME: This file is step_parser.py, function: parse model's response_id into steps, unify all modules that need steps
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch

@dataclass
class StepParseResult:
    segments: List[Dict]         # [{'role': str, 'start': int, 'end': int, 'tokens': List[int]}]
    steps: List[Dict]            # [{'action_tokens': List[int], 'observation_tokens': List[int],
                                #   'action_text': str, 'observation_text': str, 
                                #   'action_start': int, 'action_end': int, 'obs_start': int, 'obs_end': int}]
    step_ids: List[int]          # len == len(response_ids); mark k for assistant action intervals, -1 for others

def _find_first_subseq(hay, needle):
    """Safe subsequence search, avoid single token mis-matching"""
    if not needle:
        return None
    L = len(needle)
    for i in range(len(hay) - L + 1):
        if hay[i:i+L] == needle:
            return i
    return None

def _locate_template_positions(tokens: List[int], tpl: List[int]) -> List[int]:
    """Return the starting index positions where tpl appears in tokens"""
    if not tpl:  # Protection: avoid infinite loop with empty template
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
    Generic method: automatically extract role header tokens for any model
    Principle: find the role header part by comparing empty content with content-filled messages
    If extraction fails, throw an exception directly
    """
    try:
        if role == "assistant":
            # Compare differences between without assistant reply vs with assistant reply
            user_only = [{"role": "user", "content": ""}]
            user_tokens = tokenizer.apply_chat_template(
                user_only, tokenize=True, add_generation_prompt=False
            )
            
            # Complete dialog with assistant
            full_dialog = [{"role": "user", "content": ""}, {"role": "assistant", "content": "x"}]
            full_tokens = tokenizer.apply_chat_template(
                full_dialog, tokenize=True, add_generation_prompt=False
            )
            
            # Find the position of "x" (using safe subsequence search)
            x_tokens = tokenizer.encode("x", add_special_tokens=False)
            if not x_tokens:
                raise ValueError(f"Cannot encode 'x' token for role {role}")
            
            
            x_position = _find_first_subseq(full_tokens, x_tokens)
            if x_position is None:
                raise ValueError(f"Cannot find 'x' token sequence in full dialog for role {role}")
            
            
            # assistant header = from end of user_only to start of "x"
            user_len = len(user_tokens)
            
            if user_len < x_position:
                header_tokens = full_tokens[user_len:x_position]
                return header_tokens
            elif user_len == x_position:
                return []  # Return empty header, this is a valid case
            else:
                raise ValueError(f"Invalid token positions for role {role}: user_len={user_len}, x_pos={x_position}")
                
        else:
            # For user and other roles: compare empty content vs content-filled
            # Key fix: don't let user template include system message
            empty_msg = [{"role": role, "content": ""}]
            empty_tokens = tokenizer.apply_chat_template(
                empty_msg, tokenize=True, add_generation_prompt=False
            )
            
            content_msg = [{"role": role, "content": "x"}]
            content_tokens = tokenizer.apply_chat_template(
                content_msg, tokenize=True, add_generation_prompt=False
            )
            
            # Find the position of "x" (using safe subsequence search)
            x_tokens = tokenizer.encode("x", add_special_tokens=False)
            if not x_tokens:
                raise ValueError(f"Cannot encode 'x' token for role {role}")
            
            x_position = _find_first_subseq(content_tokens, x_tokens)
            if x_position is None:
                raise ValueError(f"Cannot find 'x' token sequence in content message for role {role}")
            
            # Key fix: use a more precise method to extract pure role header
            # For content_msg, the part before x should be role header
            # But if empty_tokens includes extra content (like system), need to exclude
            
            if len(content_tokens) > len(empty_tokens):
                # Added part is header + "x"
                added_part = content_tokens[len(empty_tokens):]
                x_pos_in_added = _find_first_subseq(added_part, x_tokens)
                if x_pos_in_added is not None:
                    header_tokens = added_part[:x_pos_in_added]
                else:
                    # fallback: directly take the part before x
                    header_tokens = content_tokens[:x_position]
            else:
                # directly from start to x position
                header_tokens = content_tokens[:x_position]
            
            # Additional verification: if header is too long (contains system message), try to extract pure role part
            header_decoded = tokenizer.decode(header_tokens)
            
            # If contains system message, try to take only the last role part
            if f"<|im_start|>{role}" in header_decoded:
                # Find the position of the last role marker
                role_marker = f"<|im_start|>{role}\n"
                role_tokens = tokenizer.encode(role_marker, add_special_tokens=False)
                
                # Find the position of role_tokens in header_tokens
                role_pos = _find_first_subseq(header_tokens, role_tokens)
                if role_pos is not None:
                    # Only take the role marker part
                    header_tokens = role_tokens
            return header_tokens
            
    except Exception as e:
        # Don't fallback, throw error directly
        raise RuntimeError(f"Failed to extract header tokens for role '{role}': {e}") from e

def parse_response_ids_to_steps(
    response_ids: List[int],
    tokenizer,
    assistant_tpl: List[int] = None,
    user_tpl: List[int] = None,
    mark_observation: bool = False,
) -> StepParseResult:
    # 1) Automatically extract templates
    if assistant_tpl is None:
        assistant_tpl = _extract_role_header_tokens(tokenizer, "assistant")
    if user_tpl is None:
        user_tpl = _extract_role_header_tokens(tokenizer, "user")

    # 2) Locate headers and bodies
    a_hdr = _locate_template_positions(response_ids, assistant_tpl) if assistant_tpl else []
    u_hdr = _locate_template_positions(response_ids, user_tpl) if user_tpl else []

    a_body = [p + len(assistant_tpl) for p in a_hdr] if assistant_tpl else []
    u_body = [p + len(user_tpl) for p in u_hdr] if user_tpl else []

    # If the sequence start has no headers, treat as starting from assistant content
    if response_ids:
        first_hdr = min(a_hdr[0] if a_hdr else len(response_ids),
                        u_hdr[0] if u_hdr else len(response_ids))
        if first_hdr > 0:
            a_hdr = [0] + a_hdr         # Pseudo header: for end boundary
            a_body = [0] + a_body       # Pseudo body: for start boundary

    # Use "header start" as the end boundary for splitting
    cut_bounds = sorted(a_hdr + u_hdr + [len(response_ids)])

    def next_cut(pos: int) -> int:
        for b in cut_bounds:
            if b > pos:
                return b
        return len(response_ids)

    # 3) Construct segments by body→(next header start) (won't consume next header's "user"/"assistant")
    segs = []
    for s in a_body:
        e = next_cut(s)
        if s < e:
            segs.append({"role": "assistant", "start": s, "end": e, "tokens": response_ids[s:e]})
    for s in u_body:
        e = next_cut(s)
        if s < e:
            segs.append({"role": "user", "start": s, "end": e, "tokens": response_ids[s:e]})
    segs.sort(key=lambda x: x["start"])

    if not segs:
        return StepParseResult([], [], [-1] * len(response_ids))

    # 4) Merge adjacent segments with same role
    merged = []
    for seg in segs:
        if merged and merged[-1]["role"] == seg["role"] and merged[-1]["end"] == seg["start"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["tokens"].extend(seg["tokens"])
        else:
            merged.append({
                "role": seg["role"], "start": seg["start"], "end": seg["end"],
                "tokens": seg["tokens"].copy()
            })

    # Discard segments at the beginning that are not assistant
    while merged and merged[0]["role"] != "assistant":
        merged.pop(0)
    if not merged:
        return StepParseResult([], [], [-1] * len(response_ids))

    # 5) Form steps (assistant segment + several user segments in between form observation)
    steps = []
    i = 0
    while i < len(merged):
        a = merged[i]
        if a["role"] != "assistant":
            i += 1
            continue
        action_start, action_end = a["start"], a["end"]
        action_tokens = a["tokens"]
        action_text = tokenizer.decode(action_tokens, skip_special_tokens=True)

        j = i + 1
        obs_start = action_end
        obs_end = obs_start
        obs_tokens = []
        while j < len(merged) and merged[j]["role"] != "assistant":
            obs_end = merged[j]["end"]
            obs_tokens.extend(merged[j]["tokens"])
            j += 1
        obs_text = tokenizer.decode(obs_tokens, skip_special_tokens=True) if obs_tokens else ""

        steps.append({
            "action_tokens": action_tokens,
            "observation_tokens": obs_tokens,
            "action_text": action_text,
            "observation_text": obs_text,
            "action_start": action_start, "action_end": action_end,
            "obs_start": obs_start, "obs_end": obs_end,
        })
        i = j

    # 6) Mark step_ids in place
    step_ids = [-1] * len(response_ids)
    for k, st in enumerate(steps):
        for pos in range(st["action_start"], st["action_end"]):
            step_ids[pos] = k
        if mark_observation and st["obs_start"] < st["obs_end"]:
            for pos in range(st["obs_start"], st["obs_end"]):
                step_ids[pos] = k

    return StepParseResult(merged, steps, step_ids)


# Add verification function
def verify_step_alignment(batch, tokenizer, global_step):
    """Verify step alignment between semantic evaluation and advantage scaling"""
    print(f"\n=== Step Alignment Check (Step {global_step}) ===")
    
    batch_size = len(batch.batch["prompts"])
    alignment_errors = 0
    
    for sample_idx in range(min(5, batch_size)):  # Check first 5 samples
        # Steps from semantic evaluation
        semantic_steps = batch.non_tensor_batch["steps"][sample_idx]

        # Step count from step_ids
        step_ids = batch.batch["step_ids"][sample_idx]
        max_step_id = int(step_ids.max().item()) if (step_ids >= 0).any() else -1
        advantage_steps = max_step_id + 1 if max_step_id >= 0 else 0
        
        # Check alignment
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
    """Verify consistency of step content"""
    print(f"\n=== Step Content Check (Sample {sample_idx}) ===")
    
    # Get from batch
    response_tokens = batch.batch["responses"][sample_idx].tolist()
    step_ids = batch.batch["step_ids"][sample_idx].tolist()
    semantic_steps = batch.non_tensor_batch["steps"][sample_idx]
    
    # Re-parse for verification
    from agentevolver.utils.step_parser import parse_response_ids_to_steps
    parse_result = parse_response_ids_to_steps(response_tokens, tokenizer)
    
    print(f"Parsed {len(parse_result.steps)} steps:")
    for i, step in enumerate(parse_result.steps):
        semantic_step = semantic_steps[i] if i < len(semantic_steps) else {"action": "MISSING", "observation": "MISSING"}
        print(f"Step {i}:")
        print(f"  Parsed Action: {step['action_text'][:50]}...")
        print(f"  Semantic Action: {semantic_step.get('action', 'MISSING')[:50]}...")
        print(f"  Match: {step['action_text'].strip() == semantic_step.get('action', '').strip()}")

