# -*- coding: utf-8 -*-
"""Test script for SecureMultiAgentFormatter.

Usage:
    Make sure you are in the verl conda environment:
    conda activate verl
    python games/avalon/test_secure_formatter.py
"""
import sys
from pathlib import Path
import asyncio

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from agentscope.message import Msg, TextBlock
from agentscope.token import HuggingFaceTokenCounter
from games.avalon.agents.secure_multi_agent_formatter import SecureMultiAgentFormatter


async def test_formatter():
    """Test SecureMultiAgentFormatter with truncation."""
    print("=" * 80)
    print("Testing SecureMultiAgentFormatter")
    print("=" * 80)
    
    # Create token counter (using a common model)
    try:
        token_counter = HuggingFaceTokenCounter(
            pretrained_model_name_or_path="/mnt/data/yunpeng.zyp/models/Qwen3-4B",
            use_mirror=True,
        )
        print("✓ Token counter created successfully")
    except Exception as e:
        print(f"✗ Failed to create token counter: {e}")
        return
    
    # Set a small max_tokens for testing (to trigger truncation)
    max_tokens = 200  # Small limit to test truncation
    
    # Create formatter with preserved agent names
    formatter = SecureMultiAgentFormatter(
        token_counter=token_counter,
        max_tokens=max_tokens,
        preserved_agent_names=["Moderator"],
    )
    print(f"✓ Formatter created with max_tokens={max_tokens}, preserved_agent_names=['Moderator']")
    
    # Create test messages
    messages = [
        Msg(name="Moderator", role="user", content="Game starts now. This is a moderator message."),
        Msg(name="Player0", role="user", content="Hello everyone! " * 10),  # Long message
        Msg(name="Player1", role="user", content="I agree. " * 10),  # Long message
        Msg(name="Moderator", role="user", content="Round 1 begins."),
        Msg(name="Player2", role="user", content="Let's vote. " * 10),  # Long message
        Msg(name="Player0", role="user", content="I vote yes. " * 10),  # Long message
        Msg(name="Moderator", role="user", content="Round 1 ends."),
        Msg(name="Player1", role="user", content="Round 2 starts. " * 10),  # Long message
    ]
    
    print(f"\nOriginal messages count: {len(messages)}")
    print("Message names:", [msg.name for msg in messages])
    
    # Format messages (this should trigger truncation)
    print("\nFormatting messages...")
    try:
        formatted = await formatter.format(messages)
        print(f"✓ Formatting successful")
        print(f"Formatted messages count: {len(formatted)}")
        
        # Check token count
        if token_counter:
            token_count = await formatter._count(formatted)
            print(f"Token count: {token_count} (max: {max_tokens})")
            if token_count <= max_tokens:
                print("✓ Token count is within limit")
            else:
                print("✗ Token count exceeds limit!")
        # Check if Moderator messages are preserved
        print("\nChecking if Moderator messages are preserved...")
        # Re-format to see what messages remain
        # We need to check the original messages that were kept
        # Since format() modifies the internal state, let's check the formatted output
        formatted_text = formatted[0]["content"][0]["text"] if formatted else ""
        moderator_count = formatted_text.count("<agent:Moderator>")
        print(f"Moderator messages in formatted output: {moderator_count}")
        
        if moderator_count > 0:
            print("✓ Moderator messages are preserved in formatted output")
        else:
            print("✗ No Moderator messages found in formatted output")
        
        # Print a sample of the formatted output
        print("\n" + "=" * 80)
        print("Sample formatted output (first 500 chars):")
        print("=" * 80)
        print(formatted_text[:500] + "..." if len(formatted_text) > 500 else formatted_text)
        
    except Exception as e:
        print(f"✗ Formatting failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test completed")
    print("=" * 80)


async def test_without_truncation():
    """Test formatter without truncation (no max_tokens)."""
    print("\n" + "=" * 80)
    print("Testing formatter without truncation")
    print("=" * 80)
    
    # Create formatter without max_tokens
    formatter = SecureMultiAgentFormatter(
        preserved_agent_names=["Moderator"],
    )
    print("✓ Formatter created without truncation")
    
    # Create test messages
    messages = [
        Msg(name="Moderator", role="user", content="Game starts."),
        Msg(name="Player0", role="user", content="Hello!"),
        Msg(name="Player1", role="user", content="Hi there!"),
    ]
    
    try:
        formatted = await formatter.format(messages)
        print(f"✓ Formatting successful")
        print(f"Formatted messages count: {len(formatted)}")
        
        formatted_text = formatted[0]["content"][0]["text"] if formatted else ""
        print("\nFormatted output:")
        print(formatted_text)
        
    except Exception as e:
        print(f"✗ Formatting failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_formatter())
    asyncio.run(test_without_truncation())

