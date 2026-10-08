import requests

base_url = "http://127.0.0.1:8000"

# 测试解码，示例：https%3A%2F%2Ftest.com%2Ffile%2F1
resp = requests.post(f"{base_url}/url/decode", params={"text": "https%3A%2F%2Ftest.com%2Ffile%2F1"})
print("解码结果：", resp.json())

# 测试编码
resp2 = requests.post(f"{base_url}/url/encode", params={"text": "https://test.com/file/1"})
print("编码结果：", resp2.json())
