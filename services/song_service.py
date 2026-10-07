from pydantic import BaseModel
from typing import List
from services.ollama_service import call_llava

class SongRequest(BaseModel):
    theme: str          # 主题文案
    genre: str          # 曲风：流行/说唱/国风等
    instruments: List[str] # 乐器列表
    mood: str           # 情绪基调
    length: str = "short" # short: 15-30秒小样

def generate_song(request: SongRequest):
    prompt = f"""
你是专业词曲创作人。
主题：{request.theme}
曲风：{request.genre}
情绪：{request.mood}
用到乐器：{",".join(request.instruments)}
要求：生成短版小样，适合15～30秒音频。
输出内容：
1. 歌曲名称
2. 完整歌词（分主歌、副歌）
3. 编曲说明：每个乐器的编排、段落进出时机
4. 演唱提示（语速、强弱）
不要多余废话。
"""
    # 这里对接Ollama本地大模型，传入prompt拿到结果
    result = call_llava(prompt)
    return result

# 对外暴露接口函数
def song_generator(theme, genre, instruments, mood):
    req = SongRequest(theme=theme, genre=genre, instruments=instruments, mood=mood)
    return generate_song(req)
