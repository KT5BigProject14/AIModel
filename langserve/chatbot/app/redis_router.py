# redis_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.redis_utils import save_message_to_redis, get_messages_from_redis, scan_keys, delete_message_from_redis
import uuid
from typing import Optional

router = APIRouter()


class Message(BaseModel):
    user_email: str
    session_id: Optional[str] = None
    message: str


@router.post("/save_message/")
async def save_message(msg: Message):
    if msg.session_id:
        save_message_to_redis(msg.user_email, msg.session_id, msg.message)
    else:
        session_id = str(uuid.uuid4())
        save_message_to_redis(msg.user_email, session_id, msg.message)
    return {"status": "Message saved"}


@router.get("/messages/{user_email}/{session_id}")
async def get_messages_for_user(user_email: str, session_id: str, start: int = 0, end: int = -1):
    messages = get_messages_from_redis(user_email, session_id, start, end)
    if not messages:
        raise HTTPException(status_code=404, detail="No messages found")
    return {"messages": messages}


@router.get("/all_messages/{user_email}")
async def get_all_messages_for_user(user_email: str):
    messages = scan_keys(user_email)
    return messages


@router.delete("/del_message")
async def delete_message(user_email: str, session_id: str):
    return_value = delete_message_from_redis(user_email, session_id)
    if return_value:
        return {"status": "Session deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")
