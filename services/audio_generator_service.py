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
    print(f"📥 原始返回字段：{file_path}")
    if file_path.startswith("/v1/audio"):
        # 重点：已经是完整带?path的相对地址，直接拼接BASE_URL，不要再加params！
        full_url = f"{BASE_URL}{file_path}"
        resp = requests.get(full_url)
    else:
        # 如果返回的是纯本地磁盘路径，才走params传递
        resp = requests.get(f"{BASE_URL}/v1/audio", params={"path": file_path})

    if resp.status_code != 200:
        raise Exception(f"❌ 音频接口请求失败，status={resp.status_code}, 响应：{resp.text}")

    with open(save_name, "wb") as f:
        f.write(resp.content)
    print(f"✅ 音频下载完成，本地文件：{save_name}")
    return save_name
