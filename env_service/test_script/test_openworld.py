from env_service.env_client import EnvClient



def agent_test(task_id=0):



    client = EnvClient(base_url="http://localhost:8080")

    # 获取任务列表
    env_type = "openworld"

    task_ids = client.get_env_profile(env_type, split='train')

    init_response = client.create_instance(env_type,
                                           str(task_ids[task_id]),
                                           )
    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {instance_id} with query: {query}")

    action={'role': 'assistant',
                            'content': '要判断宁德时代（CATL）的股票今天是否值得购买，我们需要分析该公司的基本面、技术面以及市场情绪等多方面的信息。由于我无法直接访问实时数据和市场动态，我们可以采取以下步骤来帮助你做出决策：\n\n1. 获取宁德时代的最新财务数据，包括但不限于营业收入、净利润、资产负债情况等。\n2. 分析最近的股价走势和技术指标。\n3. 考虑行业趋势和宏观经济环境。\n\n首先，我们可以通过调用相关工具来获取宁德时代的财务数据。请注意，这将提供一个静态的数据点，并不能单独作为投资决策的依据。我们将从中国上市公司的财务数据中获取宁德时代的信息。\n\n```json\n[\n    {\n        "tool_name": "sse_get_data_in_CHN",\n        "tool_args": {\n            "company_name": "宁德时代",\n            "year": "2022"\n        }\n    }\n]\n```\n\n以上调用将会返回宁德时代2022年的财务数据。根据这些数据，我们可以开始初步评估公司的财务状况。但请记住，进行投资之前还需要考虑其他因素，如最新的市场新闻、分析师评级及个人的投资策略等。如果你需要进一步的帮助或更详细的分析，请告诉我。'}


    result = client.step(instance_id, action)
    print(f"Step result: {result}")

    return 0





if __name__ == "__main__":
    agent_test()
