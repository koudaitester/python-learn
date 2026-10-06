import asyncio
from services.tts_service import text_to_speech

async def test_tts():
    text = "地面有水，存在滑倒安全隐患。"
    audio_path = await text_to_speech(text)
    if audio_path:
        print("✅音频生成成功，路径：", audio_path)
    else:
        print("❌音频生成失败")

if __name__ == "__main__":
    asyncio.run(test_tts())
