import pytest
import tempfile
import shutil
import os
import time
from typing import List
from unittest.mock import patch

# 导入你的实际模块
from agentevolver.client.embedding_client import OpenAIEmbeddingClient
from agentevolver.module.task_manager.strategies.deduplication.embedding import EmbeddingClient,StateRecorder,pack_trajectory


class MockTrajectory:
    """模拟的轨迹类"""
    
    def __init__(self, steps):
        self.steps = steps


class TestEmbeddingClientWithRealAPI:
    """使用真实OpenAI API的EmbeddingClient测试类"""
    
    @pytest.fixture
    def temp_db_path(self):
        """创建临时数据库路径"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def embedding_client(self, temp_db_path):
        """创建真实的EmbeddingClient"""
        # 检查API密钥是否存在
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量")
        
        from agentevolver.module.task_manager.strategies.deduplication.embedding import EmbeddingClient
        
        # 根据可用的API密钥选择配置
        if os.getenv("DASHSCOPE_API_KEY"):
            return EmbeddingClient(
                similarity_threshold=0.8,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model="text-embedding-v4",
                chroma_db_path=temp_db_path,
                collection_name="test_collection"
            )
        else:
            return EmbeddingClient(
                similarity_threshold=0.8,
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-ada-002",
                chroma_db_path=temp_db_path,
                collection_name="test_collection"
            )
    
    def test_real_embedding_initialization(self, embedding_client):
        """测试真实API的EmbeddingClient初始化"""
        assert embedding_client.similarity_threshold == 0.8
        assert embedding_client.size() == 0
        assert embedding_client.get_collection_info()["name"] == "test_collection"
    
    def test_real_add_and_retrieve(self, embedding_client):
        """测试添加文档和检索 - 使用真实API"""
        # 添加一些测试文档
        test_documents = [
            (1, "Python是一种高级编程语言"),
            (2, "机器学习是人工智能的一个分支"),
            (3, "深度学习使用神经网络"),
            (4, "数据科学涉及数据分析"),
            (5, "Web开发包括前端和后端")
        ]
        
        # 添加文档
        for doc_id, text in test_documents:
            embedding_client.add(text, doc_id)
            time.sleep(0.1)  # 避免API限流
        
        assert embedding_client.size() == 5
        
        # 测试精确匹配
        result = embedding_client.find_by_text("Python是一种高级编程语言")
        assert result == 1
        
        # 测试相似文本查找
        similar_result = embedding_client.find_by_text("Python编程语言")
        # 相似度足够高时应该找到相同的文档
        if similar_result is not None:
            assert similar_result == 1
    
    def test_real_similarity_search(self, embedding_client):
        """测试真实的相似度搜索"""
        # 添加相关文档
        programming_docs = [
            (1, "Python编程入门教程"),
            (2, "Java面向对象编程"),
            (3, "JavaScript前端开发"),
            (4, "C++系统编程"),
            (5, "机器学习算法原理")
        ]
        
        for doc_id, text in programming_docs:
            embedding_client.add(text, doc_id)
            time.sleep(0.1)
        
        # 查找与编程相关的文档
        query = "编程语言学习"
        top_results = embedding_client.find_top_k_by_text(query, k=3)
        
        assert len(top_results) <= 3
        assert len(top_results) > 0
        
        # 检查结果格式和相似度递减
        prev_similarity = 1.0
        for doc_id, similarity, text in top_results:
            assert isinstance(doc_id, int)
            assert isinstance(similarity, float)
            assert isinstance(text, str)
            assert 0 <= similarity <= 1
            assert similarity <= prev_similarity  # 相似度应该递减
            prev_similarity = similarity
            
            print(f"ID: {doc_id}, 相似度: {similarity:.3f}, 文本: {text}")
    
    def test_real_multilingual_support(self, embedding_client):
        """测试多语言支持"""
        multilingual_docs = [
            (1, "Hello world, this is a test"),
            (2, "你好世界，这是一个测试"),
            (3, "Hola mundo, esta es una prueba"),
            (4, "Bonjour le monde, c'est un test"),
            (5, "Hallo Welt, das ist ein Test")
        ]
        
        for doc_id, text in multilingual_docs:
            embedding_client.add(text, doc_id)
            time.sleep(0.1)
        
        # 测试中文查询
        chinese_result = embedding_client.find_by_text("你好世界")
        assert chinese_result == 2
        
        # 测试英文查询
        english_result = embedding_client.find_by_text("Hello world,this is a test")
        assert english_result == 1
        
        # 测试跨语言相似度
        cross_lang_results = embedding_client.find_top_k_by_text("world test", k=3)
        print("\n跨语言相似度搜索结果:")
        for doc_id, similarity, text in cross_lang_results:
            print(f"ID: {doc_id}, 相似度: {similarity:.3f}, 文本: {text}")
    
    def test_real_semantic_understanding(self, embedding_client):
        """测试语义理解能力"""
        semantic_docs = [
            (1, "汽车是一种交通工具"),
            (2, "飞机可以在天空中飞行"),
            (3, "船只在水中航行"),
            (4, "自行车需要人力驱动"),
            (5, "火车在铁轨上运行")
        ]
        
        for doc_id, text in semantic_docs:
            embedding_client.add(text, doc_id)
            time.sleep(0.1)
        
        # 测试语义相关查询
        transport_query = "交通运输方式"
        results = embedding_client.find_top_k_by_text(transport_query, k=3)
        
        print(f"\n语义搜索 '{transport_query}' 的结果:")
        for doc_id, similarity, text in results:
            print(f"ID: {doc_id}, 相似度: {similarity:.3f}, 文本: {text}")
        
        # 应该找到交通相关的文档
        assert len(results) > 0
        # 第一个结果的相似度应该相对较高
        if results:
            assert results[0][1] > 0.5  # 相似度阈值可能需要根据实际API调整
    
    def test_real_batch_processing(self, embedding_client):
        """测试批量处理"""
        # 创建大量文档
        batch_docs = []
        for i in range(20):
            batch_docs.append(f"这是第{i+1}个测试文档，内容关于批量处理测试")
        
        # 测试批量嵌入
        embeddings = embedding_client._embedding(batch_docs, bs=5)
        
        assert len(embeddings) == 20
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        
        print(f"批量处理生成了 {len(embeddings)} 个嵌入向量")
        print(f"每个向量的维度: {len(embeddings[0])}")
    
    def test_real_persistence_and_reload(self, temp_db_path):
        """测试数据持久化和重新加载"""
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量")
        
        from agentevolver.module.task_manager.strategies.deduplication.embedding import EmbeddingClient
        
        collection_name = "persistence_test"
        
        # 创建第一个客户端并添加数据
        if os.getenv("DASHSCOPE_API_KEY"):
            client1 = EmbeddingClient(
                similarity_threshold=0.8,
                chroma_db_path=temp_db_path,
                collection_name=collection_name
            )
        else:
            client1 = EmbeddingClient(
                similarity_threshold=0.8,
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-ada-002",
                chroma_db_path=temp_db_path,
                collection_name=collection_name
            )
        
        test_text = "持久化测试文档"
        client1.add(test_text, 1)
        assert client1.size() == 1
        
        # 创建第二个客户端（模拟重启）
        if os.getenv("DASHSCOPE_API_KEY"):
            client2 = EmbeddingClient(
                similarity_threshold=0.8,
                chroma_db_path=temp_db_path,
                collection_name=collection_name
            )
        else:
            client2 = EmbeddingClient(
                similarity_threshold=0.8,
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-ada-002",
                chroma_db_path=temp_db_path,
                collection_name=collection_name
            )
        
        # 数据应该仍然存在
        assert client2.size() == 1
        result = client2.find_by_text(test_text)
        assert result == 1
        
        print("持久化测试通过：数据在重启后仍然存在")


class TestStateRecorderWithRealAPI:
    """使用真实API的StateRecorder测试"""
    
    @pytest.fixture
    def temp_db_path(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def state_recorder(self, temp_db_path):
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量")
        
        from agentevolver.module.task_manager.strategies.deduplication.embedding import StateRecorder
        
        return StateRecorder(
            similarity_threshold=0.85,  # 稍高的阈值确保准确性
            chroma_db_path=temp_db_path,
            collection_name="test_states"
        )
    
    def test_real_trajectory_similarity(self, state_recorder):
        """测试真实轨迹相似度判断"""
        # 创建相似的轨迹
        trajectory1 = MockTrajectory([
            {"role": "user", "content": "我想学习Python编程"},
            {"role": "assistant", "content": "Python是一门很好的编程语言"}
        ])
        
        trajectory2 = MockTrajectory([
            {"role": "user", "content": "我想学习Python编程"},
            {"role": "assistant", "content": "Python是很好的编程语言"}
        ])
        
        # 添加第一个轨迹的状态
        state_recorder.add_state(trajectory1, "提供Python教程", "用户开始学习")
        time.sleep(0.1)
        
        # 添加第二个相似轨迹的状态
        state_recorder.add_state(trajectory2, "推荐Python资源", "用户继续学习")
        time.sleep(0.1)
        
        # 获取第一个轨迹的状态
        states1 = state_recorder.get_state(trajectory1)
        
        # 由于相似度很高，第二个轨迹应该被识别为同一个轨迹
        states2 = state_recorder.get_state(trajectory2)
        
        print(f"轨迹1的状态数量: {len(states1)}")
        print(f"轨迹2的状态数量: {len(states2)}")
        
        # 如果相似度超过阈值，两个轨迹应该共享状态
        if len(states1) == len(states2) == 2:
            print("两个相似轨迹被正确识别为同一轨迹")
        else:
            print("两个轨迹被识别为不同轨迹（可能由于相似度阈值设置）")
    
    def test_real_different_trajectories(self, state_recorder):
        """测试真实的不同轨迹处理"""
        # 创建完全不同的轨迹
        trajectory1 = MockTrajectory([
            {"role": "user", "content": "我想学习Python编程"}
        ])
        
        trajectory2 = MockTrajectory([
            {"role": "user", "content": "今天天气怎么样？"}
        ])
        
        # 添加不同轨迹的状态
        state_recorder.add_state(trajectory1, "编程指导", "学习建议")
        state_recorder.add_state(trajectory2, "天气查询", "天气信息")
        time.sleep(0.2)
        
        # 获取各自的状态
        states1 = state_recorder.get_state(trajectory1)
        states2 = state_recorder.get_state(trajectory2)
        
        assert len(states1) == 1
        assert len(states2) == 1
        assert states1[0] == ("编程指导", "学习建议")
        assert states2[0] == ("天气查询", "天气信息")
        
        print("不同轨迹的状态被正确分离")
    
    def test_real_similar_states_search(self, state_recorder):
        """测试真实的相似状态搜索"""
        # 添加多个编程相关的轨迹
        trajectories = [
            MockTrajectory([{"role": "user", "content": "Python编程入门"}]),
            MockTrajectory([{"role": "user", "content": "学习Python基础"}]),
            MockTrajectory([{"role": "user", "content": "Java编程教程"}]),
            MockTrajectory([{"role": "user", "content": "Web开发指南"}])
        ]
        
        actions = [
            "提供Python入门资料",
            "推荐Python基础教程", 
            "分享Java学习路径",
            "介绍Web开发技术"
        ]
        
        observations = [
            "用户开始Python学习",
            "用户理解Python基础",
            "用户转向Java学习",
            "用户开始Web开发"
        ]
        
        # 添加所有状态
        for traj, action, obs in zip(trajectories, actions, observations):
            state_recorder.add_state(traj, action, obs)
            time.sleep(0.1)
        
        # 查询与Python编程相关的轨迹
        query_trajectory = MockTrajectory([
            {"role": "user", "content": "Python编程学习"}
        ])
        
        similar_states = state_recorder.get_similar_states(query_trajectory, k=3)
        
        print(f"\n找到 {len(similar_states)} 个相似状态:")
        for state_id, similarity, actions_obs in similar_states:
            print(f"状态ID: {state_id}, 相似度: {similarity:.3f}")
            for action, obs in actions_obs:
                print(f"  动作: {action}, 观察: {obs}")
        
        # 应该找到一些相似的状态
        assert len(similar_states) > 0


class TestRealAPIPerformance:
    """真实API性能测试"""
    
    @pytest.fixture
    def embedding_client(self):
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量")
        
        temp_dir = tempfile.mkdtemp()
        
        from agentevolver.module.task_manager.strategies.deduplication.embedding import EmbeddingClient
        
        if os.getenv("DASHSCOPE_API_KEY"):
            client = EmbeddingClient(
                similarity_threshold=0.8,
                chroma_db_path=temp_dir,
                collection_name="perf_test"
            )
        else:
            client = EmbeddingClient(
                similarity_threshold=0.8,
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-ada-002",
                chroma_db_path=temp_dir,
                collection_name="perf_test"
            )
        
        yield client
        
        # 清理
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_api_rate_limiting(self, embedding_client):
        """测试API限流处理"""
        # 添加少量文档测试基本功能
        docs = [
            "这是第一个测试文档",
            "这是第二个测试文档", 
            "这是第三个测试文档"
        ]
        
        start_time = time.time()
        
        for i, doc in enumerate(docs):
            embedding_client.add(doc, i + 1)
            time.sleep(0.2)  # 控制请求频率
        
        elapsed_time = time.time() - start_time
        
        print(f"添加 {len(docs)} 个文档耗时: {elapsed_time:.2f} 秒")
        assert embedding_client.size() == len(docs)
    
    def test_batch_vs_individual(self, embedding_client):
        """测试批量处理vs单独处理的性能"""
        texts = [f"批量测试文档 {i}" for i in range(5)]
        
        # 测试批量嵌入
        start_time = time.time()
        batch_embeddings = embedding_client._embedding(texts, bs=3)
        batch_time = time.time() - start_time
        
        print(f"批量处理 {len(texts)} 个文档耗时: {batch_time:.2f} 秒")
        assert len(batch_embeddings) == len(texts)


# 运行配置和说明
class TestConfiguration:
    """测试配置和环境检查"""
    
    def test_environment_setup(self):
        """检查测试环境设置"""
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not dashscope_key and not openai_key:
            pytest.fail(
                "需要设置以下环境变量之一:\n"
                "- DASHSCOPE_API_KEY (阿里云DashScope)\n"
                "- OPENAI_API_KEY (OpenAI)\n\n"
                "设置方法:\n"
                "export DASHSCOPE_API_KEY='your-key-here'\n"
                "或\n"
                "export OPENAI_API_KEY='your-key-here'"
            )
        
        if dashscope_key:
            print("✅ 检测到 DashScope API Key")
        if openai_key:
            print("✅ 检测到 OpenAI API Key")


if __name__ == "__main__":
    print("🧪 EmbeddingClient 真实API测试用例")
    print("=" * 50)
    print()
    print("📋 测试内容:")
    print("✅ 真实嵌入向量生成和存储")
    print("✅ 语义相似度搜索")
    print("✅ 多语言支持测试")
    print("✅ 数据持久化验证")
    print("✅ StateRecorder轨迹管理")
    print("✅ API性能和限流测试")
    print()
    print("🔧 运行前准备:")
    print("1. 安装依赖: pip install pytest chromadb")
    print("2. 设置API密钥:")
    print("   export DASHSCOPE_API_KEY='your-dashscope-key'")
    print("   或")
    print("   export OPENAI_API_KEY='your-openai-key'")
    print()
    print("🚀 运行命令:")
    print("pytest test_embedding_client_real.py -v -s")
    print("pytest test_embedding_client_real.py::TestEmbeddingClientWithRealAPI -v")
    print()
    print("⚠️  注意:")
    print("- 测试会调用真实API，可能产生费用")
    print("- 请确保API密钥有足够的配额")
    print("- 测试中包含适当的延时以避免限流")