#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to verify httpx client cleanup in thread pool scenarios.

This script tests whether cleanup_agent_llm_clients helps prevent resource leaks
when using asyncio.run() in thread pools. Run with --with-cleanup or --without-cleanup
to test both scenarios.
"""
import asyncio
import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List

from agentscope.model import OpenAIChatModel
from agentscope.message import Msg

# Add project root to path to import games.utils
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from games.utils import cleanup_agent_llm_clients


async def single_fn_with_cleanup(thread_id: int, num_calls: int = 1) -> None:
    """Single async function that creates a model, makes calls, and cleans up."""
    print(f"[Thread {thread_id}] Starting single_fn_with_cleanup")
    
    # Create model configuration
    model_config = {
        'model_name': os.environ.get('MODEL_NAME', 'qwen-plus'),
        'client_args': {
            'base_url': os.environ.get('BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
        },
        'api_key': os.environ.get('OPENAI_API_KEY') or os.environ.get('DASHSCOPE_API_KEY', ''),
        'generate_kwargs': {
            'temperature': 0.7,
            'max_tokens': 100,
            'extra_body': {
                'enable_thinking': False,
            },
        },
        'stream': False,
    }
    
    # Create model
    model = OpenAIChatModel(**model_config)
    
    # Create a simple agent-like object with model attribute
    class SimpleAgent:
        def __init__(self, model, name):
            self.model = model
            self.name = name
    
    agent = SimpleAgent(model, f"Agent-{thread_id}")
    agents = [agent]
    
    # Make model calls
    for i in range(num_calls):
        messages = [
            {"role": "user", "content": f"Thread {thread_id}, call {i+1}: Say hello in one sentence."}
        ]
        try:
            response = await model(messages)
            print(f"[Thread {thread_id}] Call {i+1} completed: {response.content[0].text[:50] if hasattr(response.content[0], 'text') else str(response.content)[:50]}")
        except Exception as e:
            print(f"[Thread {thread_id}] Call {i+1} failed: {e}")
    
    # Clean up httpx clients
    print(f"[Thread {thread_id}] Cleaning up httpx clients...")
    await cleanup_agent_llm_clients(agents)
    print(f"[Thread {thread_id}] Cleanup completed, exiting single_fn_with_cleanup")


async def single_fn_without_cleanup(thread_id: int, num_calls: int = 1) -> None:
    """Single async function that creates a model, makes calls, but does NOT clean up."""
    print(f"[Thread {thread_id}] Starting single_fn_without_cleanup")
    
    # Create model configuration
    model_config = {
        'model_name': os.environ.get('MODEL_NAME', 'qwen-plus'),
        'client_args': {
            'base_url': os.environ.get('BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
        },
        'api_key': os.environ.get('OPENAI_API_KEY') or os.environ.get('DASHSCOPE_API_KEY', ''),
        'generate_kwargs': {
            'temperature': 0.7,
            'max_tokens': 100,
            'extra_body': {
                'enable_thinking': False,
            },
        },
        'stream': False,
    }
    
    # Create model
    model = OpenAIChatModel(**model_config)
    
    # Make model calls
    for i in range(num_calls):
        messages = [
            {"role": "user", "content": f"Thread {thread_id}, call {i+1}: Say hello in one sentence."}
        ]
        try:
            response = await model(messages)
            print(f"[Thread {thread_id}] Call {i+1} completed: {response.content[0].text[:50] if hasattr(response.content[0], 'text') else str(response.content)[:50]}")
        except Exception as e:
            print(f"[Thread {thread_id}] Call {i+1} failed: {e}")
    
    # NO cleanup - just exit
    print(f"[Thread {thread_id}] Exiting single_fn_without_cleanup (NO cleanup)")


def run_in_thread(thread_id: int, use_cleanup: bool, num_calls: int = 1) -> None:
    """Run single_fn in a thread using asyncio.run()."""
    try:
        if use_cleanup:
            asyncio.run(single_fn_with_cleanup(thread_id, num_calls))
        else:
            asyncio.run(single_fn_without_cleanup(thread_id, num_calls))
        print(f"[Thread {thread_id}] Thread completed successfully")
    except Exception as e:
        print(f"[Thread {thread_id}] Thread failed with error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Test httpx client cleanup in thread pool scenarios"
    )
    parser.add_argument(
        '--with-cleanup',
        action='store_true',
        help='Run test WITH cleanup (default: False)'
    )
    parser.add_argument(
        '--without-cleanup',
        action='store_true',
        help='Run test WITHOUT cleanup (default: False)'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=5,
        help='Number of threads in thread pool (default: 5)'
    )
    parser.add_argument(
        '--num-calls',
        type=int,
        default=5,
        help='Number of model calls per thread (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Default to both if neither is specified
    if not args.with_cleanup and not args.without_cleanup:
        print("Running both tests (with and without cleanup)")
        print("=" * 80)
        print("TEST 1: WITH CLEANUP")
        print("=" * 80)
        test_with_cleanup(args.num_threads, args.num_calls)
        print("\n" + "=" * 80)
        print("TEST 2: WITHOUT CLEANUP")
        print("=" * 80)
        test_without_cleanup(args.num_threads, args.num_calls)
    elif args.with_cleanup:
        test_with_cleanup(args.num_threads, args.num_calls)
    elif args.without_cleanup:
        test_without_cleanup(args.num_threads, args.num_calls)


def test_with_cleanup(num_threads: int, num_calls: int):
    """Run test with cleanup enabled."""
    print(f"Running test WITH cleanup: {num_threads} threads, {num_calls} calls per thread")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(run_in_thread, i, True, num_calls)
            for i in range(num_threads)
        ]
        
        # Wait for all threads to complete
        for future in futures:
            future.result()
    
    print("\nAll threads completed. Check terminal for any httpx client warnings/errors.")


def test_without_cleanup(num_threads: int, num_calls: int):
    """Run test without cleanup."""
    print(f"Running test WITHOUT cleanup: {num_threads} threads, {num_calls} calls per thread")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(run_in_thread, i, False, num_calls)
            for i in range(num_threads)
        ]
        
        # Wait for all threads to complete
        for future in futures:
            future.result()
    
    print("\nAll threads completed. Check terminal for any httpx client warnings/errors.")


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get('OPENAI_API_KEY') and not os.environ.get('DASHSCOPE_API_KEY'):
        print("Warning: OPENAI_API_KEY or DASHSCOPE_API_KEY not set in environment")
        print("Please set it before running:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("  or")
        print("  export DASHSCOPE_API_KEY='your-api-key'")
        print()
    
    main()

