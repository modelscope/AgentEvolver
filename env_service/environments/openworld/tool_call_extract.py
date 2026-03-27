import re
import json

import re
import json


def clean_pseudo_json(text: str) -> str:
    # 去注释
    text = re.sub(r'//.*|#.*', '', text)
    # 给裸字段加引号
    text = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b\s*:', r'"\1":', text)
    # 单引号换双引号
    text = text.replace("'", '"')
    # 去除尾逗号
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text.strip()


def extract_tool_calls(text: str):
    tool_calls = []

    # 1. 优先匹配 ```json [...]``` 代码块
    json_block_pattern = re.compile(r'```json\s*(\[[\s\S]*?\])\s*```', re.MULTILINE)
    matches = json_block_pattern.findall(text)

    for block in matches:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                parsed = [parsed]
            for item in parsed:
                if 'tool_name' in item and 'tool_args' in item:
                    tool_calls.append({
                        "tool_name": item["tool_name"],
                        "tool_args": item["tool_args"]
                    })
        except Exception as e:
            print(f"[JSON块解析失败]: {e}\n内容:\n{block}")

    if tool_calls:
        return tool_calls  # 找到了直接返回，优先结果

    # 2. 兜底：匹配所有方括号块，且内容中有tool_name关键字
    bracket_pattern = re.compile(r'\[\s*{[\s\S]*?}\s*\]', re.DOTALL)
    candidates = bracket_pattern.findall(text)
    for c in candidates:
        if 'tool_name' not in c:
            continue
        cleaned = clean_pseudo_json(c)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                parsed = [parsed]
            for item in parsed:
                if 'tool_name' in item and 'tool_args' in item:
                    tool_calls.append({
                        "tool_name": item["tool_name"],
                        "tool_args": item["tool_args"]
                    })
        except Exception as e:
            print(f"[伪JSON解析失败]: {e}\n内容:\n{cleaned}")

    return tool_calls


# 测试
if __name__ == "__main__":
  msg='为了帮助您判断宁德时代股票今天是否值得购买，我们可以从以下几个方面进行分析：最新的股价、涨跌幅、成交量等基础行情信息；市盈率、市净率等估值指标；以及近期的市场情绪和技术面情况。首先，我将获取宁德时代的最新行情数据。\n\n### 步骤1: 获取宁德时代的最新行情\n\n```json\n[\n  {\n    "tool_name": "tdx_PBHQInfo_quotes",\n    "tool_args": {\n      "code": "300750",\n      "setcode": "2"\n    }\n  }\n]\n```\n\n这一步骤可以帮助我们了解宁德时代的基本行情信息。接下来根据这些信息，我们可以进一步探讨其投资价值。请稍候，我将基于查询到的数据为您提供分析建议。'

  print(extract_tool_calls(msg))