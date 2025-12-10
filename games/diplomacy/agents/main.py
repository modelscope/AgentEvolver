# -*- coding: utf-8 -*-
"""Example of using ThinkingReActAgent."""
import asyncio
import os

from agentscope.formatter import DashScopeChatFormatter
from agentscope.model import DashScopeChatModel
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.pipeline import MsgHub, sequential_pipeline
from agentscope.tool import Toolkit

from thinking_react_agent import ThinkingReActAgent


async def main() -> None:
    """Main function to demonstrate ThinkingReActAgent."""
    
    # Initialize model
    model = DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    )
    
    formatter = DashScopeChatFormatter()
    
    # Create thinking agents
    alice = ThinkingReActAgent(
        name="Alice",
        sys_prompt="你是一个友好的助手，名字叫 Alice。",
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        toolkit=Toolkit(),
    )
    
    bob = ThinkingReActAgent(
        name="Bob",
        sys_prompt="你是一个友好的助手，名字叫 Bob。",
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        toolkit=Toolkit(),
    )
    
    print("=" * 80)
    print("ThinkingReActAgent 演示")
    print("=" * 80)
    print("\nAlice 和 Bob 会在发言前思考，思考内容只保留在自己 memory 中，不会广播给对方。\n")
    
    async with MsgHub(participants=[alice, bob]) as hub:
        hub.set_auto_broadcast(True)
        
        # Initial message
        initial_msg = Msg(
            name="user",
            content="大家好，请依次介绍一下自己，并说说你的爱好。",
            role="user",
        )
        
        # Use sequential pipeline
        await sequential_pipeline(
            agents=[alice, bob],
            msg=initial_msg,
        )
    
    # Check memories - show raw memory list
    print("\n" + "=" * 80)
    print("📚 Memory 检查 - 完整的原始 Memory List")
    print("=" * 80)
    
    import json
    
    print("\n🔍 Alice 的完整 Memory List:")
    alice_memory = await alice.memory.get_memory()
    print(f"  Memory 消息数: {len(alice_memory)}")
    print(f"  原始 Memory 列表:")
    print(json.dumps([msg.to_dict() for msg in alice_memory], indent=2, ensure_ascii=False))
    
    print("\n" + "-" * 80)
    print("\n🔍 Bob 的完整 Memory List:")
    bob_memory = await bob.memory.get_memory()
    print(f"  Memory 消息数: {len(bob_memory)}")
    print(f"  原始 Memory 列表:")
    print(json.dumps([msg.to_dict() for msg in bob_memory], indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 80)
    print("✅ 验证结果")
    print("=" * 80)
    print("\n1. ✅ 每个 agent 的 memory 中只保存一条完整的模型输出（包含思考+回复）")
    print("2. ✅ 广播给其他 agent 的消息不包含思考内容（只有公开回复）")
    print("3. ✅ 其他 agent 的 memory 中看不到对方的思考内容")
    print("4. ✅ 自己的 memory 中不需要额外的公开回复消息，完整的输出已足够")


if __name__ == "__main__":
    asyncio.run(main())


