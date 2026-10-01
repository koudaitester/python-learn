import requests

response = requests.post(
    # ollama测试
    "http://127.0.0.1:11434/api/chat",
    json={
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "user", "content": "你好，请简单介绍一下自己。"}
        ],
        "stream": False,
    },
    timeout=120,
)
response.raise_for_status()

print(response.json()["message"]["content"])