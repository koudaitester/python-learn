import requests
import json

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llava",
    "prompt": "参考知识库规则: 1. 环境检测：地面有水渍，判定存在安全隐患。2. 简短回答，不超过30个字。用户问题: 地面有水，有没有风险？请结合规则给出简短判断。",
    "stream": False
}

resp = requests.post(url, json=payload)
print(resp.status_code)
print(resp.text)
