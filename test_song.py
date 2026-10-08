from services.song_service import song_generator
from services.audio_generator_service import submit_music_task, poll_task_result, download_audio

if __name__ == "__main__":
    prompt = """Chinese epic ancient ballad, deep male vocal, traditional opera old-sheng tone, singing like reciting ancient poetry.
BPM 72, slow and solemn tempo.
Structure: quiet narrative verse, slowly rising into grand sweeping chorus.
Instrumentation: guzheng, pipa, erhu, large Chinese drum, stone chime.
Story background: Guan Yu of Three Kingdoms. Reads Zuo Zhuan by candlelight, bone scraping therapy, loyalty and righteousness, crossing five passes and slaying six generals, beheading Yan Liang. Turbulent Three Kingdoms era, the noble integrity of heroes.
Mood: vicissitudes, solemn, majestic, like Ode to the Red Cliff. No pop singing style, no electronic sounds.
"""
    lyrics = """【主歌1】
青烛照简编
夜读左传篇
偃月横寒刃
乱世立忠坚

刮骨谈笑间
豪气撼云天
千里寻兄路
五关斩六贤

【副歌】
丹心如日月
义薄贯长天
横刀临万阵
武圣震尘寰

烽烟分汉土
铁血铸忠言
千秋存傲骨
浩气满河山

【主歌2】
白马破狼烟
一骑斩颜良
丹心无移改
生死守盟言

鼎足三分地
英雄竞挥鞭
千秋仰高义
青史姓名传

【副歌】
丹心如日月
义薄贯长天
横刀临万阵
武圣震尘寰

烽烟分汉土
铁血铸忠言
千秋存傲骨
浩气满河山
"""
    tid = submit_music_task(prompt, lyrics, lm_temperature=0.75, lm_cfg_scale=2.4)
    result = poll_task_result(tid)
    audio_path = result["file"]
    download_audio(audio_path, "./tmp/song_audio/" + tid + ".mp3")