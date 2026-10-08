from services.song_service import song_generator
from services.audio_generator_service import submit_music_task, poll_task_result, download_audio
from services.audio_generator_service import submit_music_task, poll_task_result, download_audio

if __name__ == "__main__":
    prompt = "Chinese wuxia style, male vocal, solemn and powerful, ancient Chinese instrumental arrangement. Main instruments: pipa, guzheng, xun, shakuhachi, bamboo flute, morin khuur. The story is about Yip Man, calm and resolute martial arts master, carrying national spirit, not aggressive, steady and deep. No noisy electronic sounds."
    lyrics = """孤灯照木梁
拳藏岁月长
一身承家国
静立御风霜

寸劲破虚妄
风骨未曾忘
平凡布衣客
丹心护故乡"""
    tid = submit_music_task(prompt, lyrics, lm_temperature=0.8, lm_cfg_scale=2.2)
    result = poll_task_result(tid)
    audio_path = result["file"]
    download_audio(audio_path, "yipman_wuxia.mp3")