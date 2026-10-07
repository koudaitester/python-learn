from pydantic import BaseModel
import uuid
from typing import List

class ChatMessage(BaseModel):
    role: str # user / assistant
    content: str

# 全局内存存储
chat_sessions: dict[str, List[ChatMessage]] = {}

# 创建会话
def create_session() -> str:
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = []
    return session_id

# 追加消息
def append_msg(session_id: str, role: str, content: str):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    chat_sessions[session_id].append(ChatMessage(role=role, content=content))

# 获取历史（可限制最近N条，防止prompt超长）
def get_history(session_id: str, max_msg: int = 6) -> List[ChatMessage]:
    if session_id not in chat_sessions:
        return []
    return chat_sessions[session_id][-max_msg:]
