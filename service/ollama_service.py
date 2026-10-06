import httpx

async def call_ollama(query: str, knowledge: str, image_b64: str|None = None):
    try:
        # ollama 请求逻辑，从原来main.py剪切过来
        payload = {
            "model": "llava",
            "prompt": f"{knowledge}\n用户提问：{query}",
            "stream": False
        }
        if image_b64:
            payload["images"] = [image_b64]
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post("http://localhost:11434/api/generate", json=payload)
            data = resp.json()
            return data.get("response", "")
    except Exception as e:
        # 异常拦截，返回友好提示，不会把原始报错丢给TTS
        return "模型暂时无法响应，请稍后重试。"
