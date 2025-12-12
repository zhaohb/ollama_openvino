import json
import os

import ollama

# 禁用代理，直接连接本地 ollama 服务
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

# 创建 ollama 客户端，明确指定不使用代理
client = ollama.Client(host="http://127.0.0.1:11434")

# Load data from test.json
with open("test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Task 1: JSON format
json_prompt = f"""
{json.dumps(data, ensure_ascii=False, indent=2)}

任务：根据上述数据，找出并只返回 star 数量最多的前 3 个项目，不要返回其他信息。
请以 JSON 格式返回结果，结构如下：
{{
  "top_3_projects": [
    {{"name": "...", "repo": "...", "stars": ...}},
    {{"name": "...", "repo": "...", "stars": ...}},
    {{"name": "...", "repo": "...", "stars": ...}}
  ]
}}
"""

print("\n" + "="*60)
print("Processing ...")
print("="*60)
json_response = client.chat(
    model='Qwen3-4B-int4-asym-ov:v1',
    messages=[
        {
            'role': 'user',
            'content': json_prompt
        }
    ]
)
json_result = json_response['message']['content']
print("\nJSON format response:")
print(json_result)
