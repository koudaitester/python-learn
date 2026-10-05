import edge_tts
import os

class TTSService:
    def __init__(self, tmp_dir="./tmp"):
        self.tmp_dir = tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.audio_path = os.path.join(self.tmp_dir, "reply.mp3")
        self.voice = "zh-CN-YunyangNeural"

    async def text_to_speech(self, content: str) -> str:
        """输入文本，生成mp3音频，返回文件路径"""
        communicate = edge_tts.Communicate(content, self.voice)
        await communicate.save(self.audio_path)
        return self.audio_path

tts_service = TTSService()
