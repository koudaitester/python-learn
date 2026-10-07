from services.song_service import song_generator
from services.audio_service import generate_audio

if __name__ == "__main__":
    # 测试用例
    theme = "流感症状全部下班，支气管还在加班修复，干咳发痒"
    genre = "轻松流行说唱"
    instruments = ["木吉他", "电子鼓", "口琴"]
    mood = "调侃治愈"

    # 1. 生成词曲prompt
    song_prompt = song_generator(theme, genre, instruments, mood)
    print("===== 词曲Prompt =====")
    print(song_prompt)

    # 2. 提取歌曲名（实际项目可以用大模型单独解析title，这里先预留）
    song_title = "支气管加班日记"

    # 3. 调用音频生成，拿到文件路径
    audio_file_path = generate_audio(song_prompt, song_title)
    print(f"\n✅ 音频文件目标路径：{audio_file_path}")
