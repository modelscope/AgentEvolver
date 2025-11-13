from typing import Any, List, Dict
import torch
from loguru import logger

# apply chat_template to a message, and then convert back to message
def convert_tool_to_user_message(tool_message, format="qwen"):
    assert format == "qwen"

    if tool_message["role"] == "user":
        return tool_message
    elif tool_message["role"] == "tool" and len(tool_message["tool_calls"])>0:
        assert len(tool_message["tool_calls"])==1
        return {
            "role": "user",
            "content": str(tool_message["tool_calls"][0]['result'])
        }


def clip_state_content_correctly(tokenizer, state_content: str, max_env_len: int) -> str:
    """
    Correctly truncate state_content, ensuring token boundaries are not broken

    Args:
        tokenizer: Tokenizer
        state_content: Content to be truncated
        max_env_len: Maximum allowed token length

    Returns:
        Truncated content string
    """
    # First tokenize to check length
    tokens = tokenizer(state_content, return_tensors="pt", padding=False)["input_ids"][0]
    
    if len(tokens) <= max_env_len:
        return state_content
    
    # If too long, truncate to max_env_len length tokens
    truncated_tokens = tokens[:max_env_len]

    # Safer approach: use tokenizer's built-in methods
    # Most tokenizers have better processing methods
    if hasattr(tokenizer, 'decode'):
        # First try to preserve special tokens
        try:
            truncated_content = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
            return truncated_content
        except:
            # If failed, truncation position may be inappropriate, try removing special tokens
            try:
                truncated_content = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                return truncated_content
            except:
                # Final fallback: manual processing
                pass

    # If all decode methods fail, use a more conservative approach
    # Gradually reduce token count until successful decode
    for i in range(min(10, max_env_len)):  # Try at most 10 times
        try:
            test_tokens = tokens[:max_env_len - i]
            truncated_content = tokenizer.decode(test_tokens, skip_special_tokens=False)
            logger.warning(f"Had to reduce token count by {i} to successfully decode")
            return truncated_content
        except:
            continue

    # Final fallback: use original character truncation method
    logger.error("All token-based truncation methods failed, falling back to character truncation")
    return state_content[:max_env_len]


def get_batched_exponential_decay_weights_vectorized(
    lens: list[int],
    start_val: float = 10.0,
    end_val: float = 1.0,
    decay_reach_percent: float = 0.85,
    padding_value: float = 0.0,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Efficiently generate exponential decay weights for a batch of lengths in one go.
    This version is fully vectorized, avoiding Python loops.

    Args:
        lens (list[int]): A list containing multiple lengths.
        start_val (float): Weight value at position 0.
        end_val (float): The final value that weights decay towards.
        decay_reach_percent (float): The point where weights decay close to the final value, as a percentage of total length.
        padding_value (float): Value used to pad invalid positions.
        device: The desired device of the output tensor.

    Returns:
        torch.Tensor: A 2D weight tensor with shape (len(lens), max(lens)).
    """
    if not lens:
        return torch.empty(0, 0, device=device)

    # 1. Preparation: Get batch size, maximum length, and convert lens to tensor
    batch_size = len(lens)
    max_len = max(lens)
    lens_tensor = torch.tensor(lens, dtype=torch.float32, device=device)

    # 2. Vectorized computation of decay rate `decay_rate` for each sequence
    # Note: Each variable here is a vector with length batch_size
    amplitude = start_val - end_val
    # Subtract 1 to get the correct index range [0, length-1]
    # Use clamp(min=1) to avoid division by zero when length is 1
    decay_end_index = (lens_tensor - 1).clamp(min=1) * decay_reach_percent

    # decay_rate is a tensor with shape (batch_size,)
    decay_rate = -torch.log(torch.tensor(0.01, device=device)) / decay_end_index

    # 3. Create 2D indices and decay_rate to leverage broadcasting mechanism
    # indices shape: (max_len,) -> [0, 1, ..., max_len-1]
    indices = torch.arange(max_len, device=device)

    # decay_rate shape: (batch_size,) -> (batch_size, 1)
    # This allows it to broadcast with indices [max_len,] resulting in shape (batch_size, max_len)
    exponent = -decay_rate.unsqueeze(1) * indices

    # 4. Compute all weights at once (this is a complete BxL matrix)
    calculated_weights = amplitude * torch.exp(exponent) + end_val

    # 5. Create mask to set values at invalid positions to padding_value
    # mask shape: (batch_size, max_len)
    mask = indices < lens_tensor.unsqueeze(1)

    # 6. Apply mask to keep only weight values within valid length
    final_weights = torch.where(mask, calculated_weights, torch.tensor(padding_value, device=device))
    
    return final_weights