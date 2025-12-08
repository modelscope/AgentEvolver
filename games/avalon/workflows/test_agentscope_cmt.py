# -*- coding: utf-8 -*-
"""Test script for AgentscopeCMT class.

Usage:
    Make sure you have installed the required dependencies:
    - transformers
    - omegaconf (usually installed with the project)
    
    Then run:
    python games/avalon/workflows/test_agentscope_cmt.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

# Add workspace root to path to import modules
workspace_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(workspace_root))

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Please install it first.")
    sys.exit(1)

try:
    from games.avalon.workflows.agentscope_cmt import AgentscopeCMT
    from agentevolver.schema.trajectory import Reward
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure you are in the project root directory and dependencies are installed.")
    sys.exit(1)


class MockConfig:
    """Mock config object for testing."""
    def __init__(self):
        self.actor_rollout_ref = SimpleNamespace()
        self.actor_rollout_ref.rollout = SimpleNamespace()
        self.actor_rollout_ref.rollout.response_length = 2048
        self.actor_rollout_ref.rollout.max_model_len = 4096
        self.actor_rollout_ref.rollout.max_env_len = 2000
        
        self.data = SimpleNamespace()
        self.data.max_prompt_length = 2048
        self.data.max_response_length = 2048


def create_mock_config():
    """Create a mock config object for testing."""
    return MockConfig()


def test_agentscope_cmt_basic():
    """Test basic functionality of AgentscopeCMT."""
    print("=" * 80)
    print("Test 1: Basic functionality")
    print("=" * 80)
    
    # Create mock config and tokenizer
    config = create_mock_config()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    
    # Create mock model_call_history
    model_call_history = [
        {
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            "response": "2+2 equals 4."
        },
        {
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "response": "The capital of France is Paris."
        }
    ]
    
    # Create AgentscopeCMT instance
    cmt = AgentscopeCMT(
        config=config,
        tokenizer=tokenizer,
        model_call_history=model_call_history,
        reward=Reward(outcome=1.0, success_rate=1.0),
        data_id="test_data_001",
        rollout_id="test_rollout_001",
        task_id="test_task_001"
    )
    
    # Test group_tokenize
    sample_arr = cmt.group_tokenize()
    
    # Assertions
    assert len(sample_arr) == 2, f"Expected 2 samples, got {len(sample_arr)}"
    print(f"✅ Correct number of samples: {len(sample_arr)}")
    
    for i, sample in enumerate(sample_arr):
        assert sample.data_id == "test_data_001", f"Sample {i}: Wrong data_id"
        assert sample.rollout_id == "test_rollout_001", f"Sample {i}: Wrong rollout_id"
        assert sample.task_id == "test_task_001", f"Sample {i}: Wrong task_id"
        assert sample.minor_index_id == i, f"Sample {i}: Wrong minor_index_id"
        print(f"✅ Sample {i}: IDs are correct")
    
    print("✅ Test 1 passed!\n")


def test_loss_mask_correctness():
    """Test that loss_mask is correct: only response tokens should have mask=1."""
    print("=" * 80)
    print("Test 2: Loss mask correctness")
    print("=" * 80)
    
    # Create mock config and tokenizer
    config = create_mock_config()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    
    # Create simple model_call_history
    model_call_history = [
        {
            "prompt": [
                {"role": "user", "content": "Hello"}
            ],
            "response": "Hi there!"
        }
    ]
    
    # Create AgentscopeCMT instance
    cmt = AgentscopeCMT(
        config=config,
        tokenizer=tokenizer,
        model_call_history=model_call_history,
        reward=Reward(outcome=1.0),
        data_id="test_data_002",
        rollout_id="test_rollout_002",
        task_id="test_task_002"
    )
    
    # Test group_tokenize
    sample_arr = cmt.group_tokenize()
    assert len(sample_arr) == 1, "Expected 1 sample"
    
    sample = sample_arr[0]
    
    # Check loss_mask
    loss_mask = sample.loss_mask
    prompt_ids = sample.prompt_ids
    response_ids = sample.response_ids
    
    print(f"Prompt length: {len(prompt_ids)}")
    print(f"Response length: {len(response_ids)}")
    print(f"Total loss_mask length: {len(loss_mask)}")
    print(f"Prompt loss_mask length: {len(sample.prompt_loss_mask)}")
    print(f"Response loss_mask length: {len(sample.response_loss_mask)}")
    
    # Check that prompt_loss_mask is all zeros
    prompt_loss_sum = sum(sample.prompt_loss_mask)
    assert prompt_loss_sum == 0, f"Prompt loss_mask should be all zeros, but sum is {prompt_loss_sum}"
    print(f"✅ Prompt loss_mask is all zeros (sum={prompt_loss_sum})")
    
    # Check that response_loss_mask has some ones (response should participate in training)
    response_loss_sum = sum(sample.response_loss_mask)
    assert response_loss_sum > 0, f"Response loss_mask should have some ones, but sum is {response_loss_sum}"
    print(f"✅ Response loss_mask has training tokens (sum={response_loss_sum})")
    
    # Verify the split is correct
    assert len(prompt_ids) == len(sample.prompt_loss_mask), "Prompt IDs and loss_mask length mismatch"
    assert len(response_ids) == len(sample.response_loss_mask), "Response IDs and loss_mask length mismatch"
    print("✅ Loss mask lengths match token lengths")
    
    print("✅ Test 2 passed!\n")


def test_multiple_samples():
    """Test that each prompt-response pair generates a separate sample."""
    print("=" * 80)
    print("Test 3: Multiple samples from multiple call records")
    print("=" * 80)
    
    # Create mock config and tokenizer
    config = create_mock_config()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    
    # Create multiple model_call_history entries
    model_call_history = [
        {
            "prompt": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "Calculate 5 * 3"}
            ],
            "response": "5 * 3 = 15"
        },
        {
            "prompt": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 10 / 2?"}
            ],
            "response": "10 / 2 = 5"
        },
        {
            "prompt": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "Solve 2 + 2"}
            ],
            "response": "2 + 2 = 4"
        }
    ]
    
    # Create AgentscopeCMT instance
    cmt = AgentscopeCMT(
        config=config,
        tokenizer=tokenizer,
        model_call_history=model_call_history,
        reward=Reward(outcome=1.0),
        data_id="test_data_003",
        rollout_id="test_rollout_003",
        task_id="test_task_003"
    )
    
    # Test group_tokenize
    sample_arr = cmt.group_tokenize()
    
    assert len(sample_arr) == 3, f"Expected 3 samples, got {len(sample_arr)}"
    print(f"✅ Generated {len(sample_arr)} samples from {len(model_call_history)} call records")
    
    # Check each sample is independent
    for i, sample in enumerate(sample_arr):
        assert sample.minor_index_id == i, f"Sample {i}: Wrong minor_index_id"
        
        # Each sample should have its own prompt and response
        assert len(sample.prompt_ids) > 0, f"Sample {i}: Prompt should not be empty"
        assert len(sample.response_ids) > 0, f"Sample {i}: Response should not be empty"
        
        # Check loss_mask correctness for each sample
        prompt_loss_sum = sum(sample.prompt_loss_mask)
        response_loss_sum = sum(sample.response_loss_mask)
        
        assert prompt_loss_sum == 0, f"Sample {i}: Prompt loss_mask should be all zeros"
        assert response_loss_sum > 0, f"Sample {i}: Response loss_mask should have some ones"
        
        print(f"  Sample {i}: prompt_tokens={len(sample.prompt_ids)}, "
              f"response_tokens={len(sample.response_ids)}, "
              f"response_trainable_tokens={response_loss_sum}")
    
    print("✅ Test 3 passed!\n")


def test_tokenization_consistency():
    """Test that tokenization is consistent and correct."""
    print("=" * 80)
    print("Test 4: Tokenization consistency")
    print("=" * 80)
    
    # Create mock config and tokenizer
    config = create_mock_config()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    
    # Create model_call_history
    model_call_history = [
        {
            "prompt": [
                {"role": "user", "content": "Say hello"}
            ],
            "response": "Hello! How can I help you?"
        }
    ]
    
    # Create AgentscopeCMT instance
    cmt = AgentscopeCMT(
        config=config,
        tokenizer=tokenizer,
        model_call_history=model_call_history,
        reward=Reward(outcome=1.0),
        data_id="test_data_004",
        rollout_id="test_rollout_004",
        task_id="test_task_004"
    )
    
    # Test group_tokenize
    sample_arr = cmt.group_tokenize()
    assert len(sample_arr) == 1, "Expected 1 sample"
    
    sample = sample_arr[0]
    
    # Check that input_ids = prompt_ids + response_ids
    assert len(sample.input_ids) == len(sample.prompt_ids) + len(sample.response_ids), \
        "input_ids should be concatenation of prompt_ids and response_ids"
    print(f"✅ input_ids length ({len(sample.input_ids)}) = prompt_ids ({len(sample.prompt_ids)}) + response_ids ({len(sample.response_ids)})")
    
    # Check that loss_mask = prompt_loss_mask + response_loss_mask
    assert len(sample.loss_mask) == len(sample.prompt_loss_mask) + len(sample.response_loss_mask), \
        "loss_mask should be concatenation of prompt_loss_mask and response_loss_mask"
    print(f"✅ loss_mask length ({len(sample.loss_mask)}) = prompt_loss_mask ({len(sample.prompt_loss_mask)}) + response_loss_mask ({len(sample.response_loss_mask)})")
    
    # Verify the actual concatenation
    expected_input_ids = sample.prompt_ids + sample.response_ids
    assert sample.input_ids == expected_input_ids, "input_ids should match concatenation"
    print("✅ input_ids matches concatenation of prompt_ids and response_ids")
    
    expected_loss_mask = sample.prompt_loss_mask + sample.response_loss_mask
    assert sample.loss_mask == expected_loss_mask, "loss_mask should match concatenation"
    print("✅ loss_mask matches concatenation of prompt_loss_mask and response_loss_mask")
    
    print("✅ Test 4 passed!\n")


def test_empty_response():
    """Test handling of empty response."""
    print("=" * 80)
    print("Test 5: Empty response handling")
    print("=" * 80)
    
    # Create mock config and tokenizer
    config = create_mock_config()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    
    # Create model_call_history with empty response
    model_call_history = [
        {
            "prompt": [
                {"role": "user", "content": "Test"}
            ],
            "response": ""
        }
    ]
    
    # Create AgentscopeCMT instance
    cmt = AgentscopeCMT(
        config=config,
        tokenizer=tokenizer,
        model_call_history=model_call_history,
        reward=Reward(outcome=0.0),
        data_id="test_data_005",
        rollout_id="test_rollout_005",
        task_id="test_task_005"
    )
    
    # Test group_tokenize
    sample_arr = cmt.group_tokenize()
    assert len(sample_arr) == 1, "Expected 1 sample even with empty response"
    
    sample = sample_arr[0]
    # Empty response should still create a sample, but response_ids might be minimal
    print(f"✅ Sample created with empty response: response_tokens={len(sample.response_ids)}")
    
    print("✅ Test 5 passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("AgentscopeCMT Test Suite")
    print("=" * 80 + "\n")
    
    try:
        test_agentscope_cmt_basic()
        test_loss_mask_correctness()
        test_multiple_samples()
        test_tokenization_consistency()
        test_empty_response()
        
        print("=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

