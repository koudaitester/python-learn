import requests
import json

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava"

def call_llava(prompt: str, image_base64: str | None):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    if image_base64 is not None:
        payload["images"] = [image_base64]

    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        print("status_code:", resp.status_code)
        print("raw response text:", repr(resp.text))
        data = resp.json()
        return data["response"]
    except json.JSONDecodeError:
        return "模型加载失败，Ollama模型进程意外终止。"
    except Exception as e:
        print(f"调用异常：{str(e)}")
        return f"模型调用异常：{str(e)}"
