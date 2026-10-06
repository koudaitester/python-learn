import uuid
from pathlib import Path
# 这里导入你原来的TTS核心包，比如 edge-tts 或者其他
import edge_tts

# 音频输出目录
TMP_AUDIO_DIR = Path("./tmp")
TMP_AUDIO_DIR.mkdir(exist_ok=True)

async def text_to_speech(text: str) -> str:
    """
    输入文本，生成音频，返回音频文件绝对路径
    """
    try:
        # uuid生成唯一文件名，防止文件互相覆盖
        file_name = f"{uuid.uuid4()}.mp3"
        audio_path = str(TMP_AUDIO_DIR / file_name)

        # 这里用你原来的TTS配置，替换成你在用的语音角色
        communicate = edge_tts.Communicate(text, voice="zh-CN-YunyangNeural")
        await communicate.save(audio_path)
        return audio_path
    except Exception as e:
        print(f"TTS生成失败: {e}")
        return ""
