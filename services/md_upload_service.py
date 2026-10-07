import os
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
STORE_DIR = BASE_DIR / "md"
STORE_DIR.mkdir(exist_ok=True, parents=True)

ALLOW_SUFFIX = {".md"}

def check_md_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOW_SUFFIX

def save_md_file(file_bytes: bytes, original_filename: str) -> dict:
    if not check_md_file(original_filename):
        raise ValueError("仅支持上传 .md 文件")
    # 生成唯一id，防止同名覆盖
    file_id = str(uuid.uuid4())
    suffix = Path(original_filename).suffix
    save_name = f"{file_id}{suffix}"
    save_path = STORE_DIR / save_name
    save_path.write_bytes(file_bytes)

    return {
        "file_id": file_id,
        "original_name": original_filename,
        "save_path": str(save_path),
        "size_bytes": len(file_bytes)
    }
