from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    image_base64: str | None = None
