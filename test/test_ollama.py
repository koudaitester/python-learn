import asyncio
from services.ollama_service import call_llava

async def test_ollama():
    prompt = """参考知识库规则:
【知识库规则】
1. 环境检测：地面有水渍，判定存在安全隐患。
2. 简短回答，不超过30个字。
用户问题: 地面有水，有没有风险？
请结合规则给出简短判断。
"""
    result = call_llava(prompt, None)
    print("====模型返回结果====")
    print(result)

if __name__ == "__main__":
    asyncio.run(test_ollama())
