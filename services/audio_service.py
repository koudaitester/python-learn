import os
import time
import uuid

# 音频输出根目录
AUDIO_OUTPUT_DIR = "./tmp/song_audio"

def init_audio_dir():
    """初始化音频存储目录，不存在则创建"""
    if not os.path.exists(AUDIO_OUTPUT_DIR):
        os.makedirs(AUDIO_OUTPUT_DIR)

def generate_audio_filename(song_title: str) -> str:
    """
    生成安全的音频文件名
    格式：{歌曲名}_{时间戳}_{唯一id}.wav
    """
    # 清洗文件名，去除非法字符
    safe_title = "".join(c for c in song_title if c.isalnum() or c in (" ", "_")).strip()
    timestamp = int(time.time())
    short_id = str(uuid.uuid4())[:8]
    filename = f"{safe_title}_{timestamp}_{short_id}.wav"
    return os.path.join(AUDIO_OUTPUT_DIR, filename)

def generate_audio(song_prompt: str, song_title: str):
    """
    预留音频生成接口
    这里后续接入音频生成模型API/本地音乐模型
    返回生成好的音频文件完整路径
    """
    init_audio_dir()
    full_path = generate_audio_filename(song_title)
    
    # ========== 这里是预留占位 ==========
    # 后续在这里调用音乐生成模型，输出音频写入 full_path
    # 示例：music_model.generate(prompt=song_prompt, save_path=full_path)
    # =====================================
    print(f"【占位】音频将保存至：{full_path}")
    return full_path
