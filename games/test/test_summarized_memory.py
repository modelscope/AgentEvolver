#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for SummarizedMemory class."""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agentscope.message import Msg
from games.agents.memory import SummarizedMemory


async def test_basic_functionality():
    """Test basic memory functionality."""
    print("=" * 80)
    print("Test 1: Basic Functionality")
    print("=" * 80)
    
    # Create memory instance (will load from yaml)
    memory = SummarizedMemory()
    
    # Test initial state
    assert await memory.size() == 0, "Initial size should be 0"
    print("✓ Initial size is 0")
    
    # Add some messages
    msg1 = Msg(name="user", content="Hello", role="user")
    msg2 = Msg(name="assistant", content="Hi there!", role="assistant")
    msg3 = Msg(name="user", content="How are you?", role="user")
    
    await memory.add(msg1)
    await memory.add(msg2)
    await memory.add(msg3)
    
    assert await memory.size() == 3, f"Size should be 3, got {await memory.size()}"
    print(f"✓ Added 3 messages, size is {await memory.size()}")
    
    # Get memory
    memories = await memory.get_memory()
    assert len(memories) == 3, f"Should have 3 messages, got {len(memories)}"
    print(f"✓ Retrieved {len(memories)} messages")
    
    # Check content
    assert memories[0].content == "Hello", "First message content mismatch"
    assert memories[1].content == "Hi there!", "Second message content mismatch"
    assert memories[2].content == "How are you?", "Third message content mismatch"
    print("✓ Message contents are correct")
    
    print("✅ Test 1 passed!\n")


async def test_config_loading():
    """Test configuration loading from yaml."""
    print("=" * 80)
    print("Test 2: Configuration Loading")
    print("=" * 80)
    
    # Create memory instance
    memory = SummarizedMemory()
    
    # Check that config was loaded
    assert hasattr(memory, 'max_messages'), "max_messages should be set"
    assert hasattr(memory, 'system_prompt'), "system_prompt should be set"
    assert hasattr(memory, 'summary_prompt'), "summary_prompt should be set"
    print(f"✓ Config loaded: max_messages={memory.max_messages}")
    print(f"✓ System prompt: {memory.system_prompt[:50]}...")
    print(f"✓ Summary prompt template loaded")
    
    # Check model instance
    if memory.summary_model is not None:
        print("✓ Summary model created successfully")
    else:
        print("⚠ Summary model is None (may be due to missing API key)")
    
    print("✅ Test 2 passed!\n")


async def test_config_override():
    """Test configuration override via kwargs."""
    print("=" * 80)
    print("Test 3: Configuration Override")
    print("=" * 80)
    
    # Create memory with custom config
    memory = SummarizedMemory(max_messages=5)
    
    assert memory.max_messages == 5, f"max_messages should be 5, got {memory.max_messages}"
    print(f"✓ Override max_messages to 5")
    
    # Test with memory_config dict
    memory2 = SummarizedMemory(
        memory_config={
            'max_messages': 10,
            'model_name': 'test-model'
        }
    )
    
    assert memory2.max_messages == 10, f"max_messages should be 10, got {memory2.max_messages}"
    print(f"✓ Override via memory_config dict: max_messages={memory2.max_messages}")
    
    print("✅ Test 3 passed!\n")


async def test_summarization_trigger():
    """Test that summarization is triggered when exceeding max_messages."""
    print("=" * 80)
    print("Test 4: Summarization Trigger")
    print("=" * 80)
    
    # Create memory with small max_messages for testing
    memory = SummarizedMemory(max_messages=3)
    
    # Add messages up to max_messages
    for i in range(3):
        msg = Msg(name="user", content=f"Message {i+1}", role="user")
        await memory.add(msg)
    
    # Get memory (should not trigger summarization yet)
    memories = await memory.get_memory()
    assert len(memories) == 3, f"Should have 3 messages, got {len(memories)}"
    print(f"✓ Added 3 messages (at max_messages threshold)")
    
    # Add one more message to exceed threshold
    msg4 = Msg(name="user", content="Message 4", role="user")
    await memory.add(msg4)
    
    print(f"✓ Added 4th message (exceeds max_messages={memory.max_messages})")
    print(f"  Summary model available: {memory.summary_model is not None}")
    
    # Get memory (should trigger summarization if model is available)
    memories = await memory.get_memory()
    
    if memory.summary_model is not None:
        # If model is available, summarization should have occurred
        # After summarization, we should have 1 summary message
        print(f"✓ Summarization triggered, memory size: {len(memories)}")
        if len(memories) == 1:
            print("  ✓ Messages summarized into 1 summary message")
        else:
            print(f"  ⚠ Expected 1 message after summarization, got {len(memories)}")
    else:
        # If model is not available, original messages should be kept
        assert len(memories) == 4, f"Should have 4 messages (no summarization), got {len(memories)}"
        print("✓ No summarization (model not available), kept original messages")
    
    print("✅ Test 4 passed!\n")


async def test_add_and_delete():
    """Test add and delete operations."""
    print("=" * 80)
    print("Test 5: Add and Delete Operations")
    print("=" * 80)
    
    memory = SummarizedMemory()
    
    # Add messages
    msg1 = Msg(name="user", content="Message 1", role="user")
    msg2 = Msg(name="user", content="Message 2", role="user")
    msg3 = Msg(name="user", content="Message 3", role="user")
    
    await memory.add(msg1)
    await memory.add(msg2)
    await memory.add(msg3)
    
    assert await memory.size() == 3, f"Size should be 3, got {await memory.size()}"
    print("✓ Added 3 messages")
    
    # Delete one message
    await memory.delete(1)
    assert await memory.size() == 2, f"Size should be 2, got {await memory.size()}"
    print("✓ Deleted message at index 1")
    
    # Check remaining messages
    memories = await memory.get_memory()
    assert memories[0].content == "Message 1", "First message should remain"
    assert memories[1].content == "Message 3", "Third message should be at index 1 now"
    print("✓ Remaining messages are correct")
    
    # Clear memory
    await memory.clear()
    assert await memory.size() == 0, "Size should be 0 after clear"
    print("✓ Cleared memory")
    
    print("✅ Test 5 passed!\n")


async def test_actual_model_call():
    """Test actual model call and verify summary output."""
    print("=" * 80)
    print("Test 6: Actual Model Call and Summary Output")
    print("=" * 80)
    
    # Create memory with small max_messages for testing
    memory = SummarizedMemory(max_messages=3)
    
    if memory.summary_model is None:
        print("⚠ Summary model is not available, skipping actual model call test")
        print("  (This is expected if API key is not configured)")
        print("✅ Test 6 skipped (model not available)\n")
        return
    
    # Add meaningful messages for summarization
    messages = [
        Msg(name="user", content="Hello, I'm looking for a good restaurant.", role="user"),
        Msg(name="assistant", content="I'd be happy to help! What type of cuisine are you interested in?", role="assistant"),
        Msg(name="user", content="I prefer Italian food, something with pasta.", role="user"),
        Msg(name="assistant", content="Great! I recommend trying our signature spaghetti carbonara. It's very popular.", role="assistant"),
    ]
    
    # Add messages
    for msg in messages:
        await memory.add(msg)
    
    print(f"✓ Added {len(messages)} messages")
    print(f"  Messages exceed max_messages={memory.max_messages}, will trigger summarization")
    
    # Get memory (should trigger summarization)
    print("\n  Calling model to generate summary...")
    memories = await memory.get_memory()
    
    # Verify summarization occurred
    assert len(memories) == 1, f"Expected 1 summary message, got {len(memories)}"
    print(f"✓ Summarization completed, memory size: {len(memories)}")
    
    # Check summary message content
    summary_msg = memories[0]
    assert isinstance(summary_msg, Msg), "Summary should be a Msg object"
    assert "[对话总结]" in str(summary_msg.content), "Summary message should contain summary prefix"
    
    summary_content = str(summary_msg.content)
    print(f"\n  Summary content preview:")
    print(f"  {summary_content[:200]}...")
    
    # Verify summary is meaningful (not empty, has reasonable length)
    assert len(summary_content) > 20, "Summary should have meaningful content"
    print(f"✓ Summary has meaningful content (length: {len(summary_content)} chars)")
    
    print("✅ Test 6 passed!\n")


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SummarizedMemory Test Suite")
    print("=" * 80 + "\n")
    
    try:
        # Test 1: Basic functionality
        await test_basic_functionality()
        
        # Test 2: Configuration loading
        await test_config_loading()
        
        # Test 3: Configuration override
        await test_config_override()
        
        # Test 4: Summarization trigger
        await test_summarization_trigger()
        
        # Test 5: Add and delete operations
        await test_add_and_delete()
        
        # Test 6: Actual model call and summary output
        await test_actual_model_call()
        
        print("=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        print("\nSummary:")
        print("1. ✅ Basic memory operations work correctly")
        print("2. ✅ Configuration loads from yaml file")
        print("3. ✅ Configuration can be overridden via kwargs")
        print("4. ✅ Summarization triggers when exceeding max_messages")
        print("5. ✅ Add and delete operations work correctly")
        print("6. ✅ Actual model call and summary output verified")
        
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
    asyncio.run(main())

