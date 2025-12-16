#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to check what agentscope model call returns."""
import asyncio
import json
import os
from pprint import pprint

from agentscope.model import OpenAIChatModel
from agentscope.message import Msg


async def test_model_call():
    """Test OpenAIChatModel call and inspect the return value."""
    
    # Configuration from task_config.yaml
    model_config = {
        'model_name': 'qwen3-32b',  # or 'qwen3-32b' from default_model
        'client_args': {
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        },
        'api_key': os.environ.get('OPENAI_API_KEY', ''),
        'generate_kwargs': {
            'temperature': 0.7,
            'max_tokens': 2048,
            'extra_body': {
                'enable_thinking': False,  # Required for non-streaming calls with DashScope
            },
        },
        'stream': False,  # Set to False for easier inspection
    }
    
    print("=" * 80)
    print("Testing OpenAIChatModel with qwen2.5-32b-instruct")
    print("=" * 80)
    print(f"Model config: {json.dumps(model_config, indent=2, ensure_ascii=False)}")
    print()
    
    # Create model
    model = OpenAIChatModel(**model_config)
    
    # Prepare test messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello! Please introduce yourself briefly."
        }
    ]
    
    print("=" * 80)
    print("Input messages:")
    print("=" * 80)
    pprint(messages)
    print()
    
    # Call the model
    print("=" * 80)
    print("Calling model...")
    print("=" * 80)
    response = await model(messages)
    
    # Inspect the response
    print("=" * 80)
    print("Response Type:")
    print("=" * 80)
    print(f"Type: {type(response)}")
    print(f"Type name: {type(response).__name__}")
    print()
    
    print("=" * 80)
    print("Response Object:")
    print("=" * 80)
    print(f"Response: {response}")
    print()
    
    print("=" * 80)
    print("Response Attributes:")
    print("=" * 80)
    if hasattr(response, '__dict__'):
        print("__dict__:")
        pprint(response.__dict__)
    print()
    
    # Check if it's a ChatResponse
    if hasattr(response, 'content'):
        print("=" * 80)
        print("Response.content:")
        print("=" * 80)
        print(f"Type: {type(response.content)}")
        print(f"Content: {response.content}")
        print()
        
        # Inspect content blocks
        if isinstance(response.content, (list, tuple)):
            print("=" * 80)
            print("Content Blocks:")
            print("=" * 80)
            for i, block in enumerate(response.content):
                print(f"\nBlock {i}:")
                print(f"  Type: {type(block)}")
                print(f"  Block: {block}")
                if hasattr(block, '__dict__'):
                    print(f"  Attributes: {block.__dict__}")
        else:
            print(f"Content (not a list): {response.content}")
        print()
    
    if hasattr(response, 'usage'):
        print("=" * 80)
        print("Response.usage:")
        print("=" * 80)
        print(f"Usage: {response.usage}")
        if hasattr(response.usage, '__dict__'):
            print(f"Usage attributes: {response.usage.__dict__}")
        print()
    
    if hasattr(response, 'metadata'):
        print("=" * 80)
        print("Response.metadata:")
        print("=" * 80)
        print(f"Metadata: {response.metadata}")
        print()
    
    if hasattr(response, 'id'):
        print("=" * 80)
        print("Response.id:")
        print("=" * 80)
        print(f"ID: {response.id}")
        print()
    
    if hasattr(response, 'created_at'):
        print("=" * 80)
        print("Response.created_at:")
        print("=" * 80)
        print(f"Created at: {response.created_at}")
        print()
    


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get('DASHSCOPE_API_KEY'):
        print("Warning: DASHSCOPE_API_KEY not set in environment")
        print("Please set it before running:")
        print("  export DASHSCOPE_API_KEY='your-api-key'")
        print()
    
    # Run the test
    asyncio.run(test_model_call())

