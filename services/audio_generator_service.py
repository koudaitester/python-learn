import requests
import time
import json
import shutil
import re

BASE_URL = "http://127.0.0.1:8001"

def submit_music_task(prompt: str, lyrics: str, lm_temperature: float, lm_cfg_scale: float):
    resp = requests.post(f"{BASE_URL}/release_task", json={
        "prompt": prompt,
        "lyrics": lyrics,
        "thinking": True,
        "lm_temperature": lm_temperature,
        "lm_cfg_scale": lm_cfg_scale
    })
    task_id = resp.json()["data"]["task_id"]
    print(f"✅ 已提交音乐任务，task_id = {task_id}")
    return task_id

def poll_task_result(task_id: str, interval=20):
    print(f"🔍 开始轮询任务状态，task_id = {task_id}")
    while True:
        resp = requests.post(f"{BASE_URL}/query_result", json={"task_id_list": [task_id]})
        outer_data = resp.json()["data"][0]
        result_list = json.loads(outer_data["result"])
        res = result_list[0]

        progress = res["progress"]
        stage = res["stage"]
        print(f"📊 task_id:{task_id} | 进度：{progress:.2f} | 阶段：{stage}")

        if res["status"] == 1:
            print(f"🎉 任务完成！获取音频文件路径：{res['file']}")
            return res
        elif res["status"] == -1:
            raise Exception(f"❌ 任务失败：{res}")
        time.sleep(interval)

def download_audio(file_path: str, save_name="output.mp3"):
    print(f"📥 准备下载音频，远端路径：{file_path}，本地保存名：{save_name}")
    resp = requests.get(f"{BASE_URL}/v1/audio", params={"path": file_path})
    with open(save_name, "wb") as f:
        f.write(resp.content)
    print(f"✅ 音频下载完成，本地文件：{save_name}")
    return save_name

def copy_audio_directly(file_path: str, save_name="output.mp3"):
    print(f"📂 原始返回字段: {file_path}")
    # 提取path=后面的本地磁盘路径
    match = re.search(r"path=(.+)", file_path)
    if match:
        source_path = match.group(1)
    else:
        source_path = file_path
    print(f"📂 源文件路径: {source_path}")
    shutil.copy(source_path, save_name)
    print(f"✅ 文件拷贝完成，本地音频：{save_name}")
    return save_name