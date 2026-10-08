from services.song_service import song_generator
from services.audio_generator_service import copy_audio_directly, submit_music_task, poll_task_result, download_audio

if __name__ == "__main__":
    prompt = """Chinese martial arts song, male vocal, clear intelligible vocals.
Song structure: calm intro, gradually build up, strong fast-paced climax chorus, obvious rhythm contrast.
Instrumentation: guzheng, pipa, erhu, Chinese drum, bamboo flute.
Story background: Bruce Lee, founder of Jeet Kune Do, inch punch, no-limits combat philosophy. Cross-cultural kung fu icon, spread Chinese martial arts to the world, break cultural barriers.
Mood: verse is calm and thoughtful, chorus powerful, energetic, punchy rhythm. No electronic music.
"""
    lyrics = """【主歌1】
海面起长风
拳藏万象中
不困门派笼
破尽旧樊笼

寸劲一瞬涌
动静自相通
武道非争勇
心与天地同

【副歌】
截拳破虚锋
一念定苍穹
以武载大道
四海识龙踪

鼓震山河动
傲骨贯长虹
身传华夏意
万里振雄风

【主歌2】
银幕展真容
风骨映苍穹
打破东西隔
武道入寰中

不拘招式重
顺势化千攻
平生持正念
侠气贯始终

【副歌】
截拳破虚锋
一念定苍穹
以武载大道
四海识龙踪

鼓震山河动
傲骨贯长虹
身传华夏意
万里振雄风
"""
    tid = submit_music_task(prompt, lyrics, lm_temperature=0.75, lm_cfg_scale=2.4)
    result = poll_task_result(tid)
    audio_path = result["file"]
    copy_audio_directly(audio_path, "./tmp/song_audio/" + tid + ".mp3")