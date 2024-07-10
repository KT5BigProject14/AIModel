# redis_utils.py
from core.redis_config import redis_conn

from pydantic import BaseModel
from typing import List


def get_redis_session_key(user_email: str, session_id: str) -> str:
    return f"user:{user_email}:session:{session_id}:messages"


def save_message_to_redis(user_email: str, session_id: str, message: str):
    key = get_redis_session_key(user_email, session_id)
    redis_conn.lpush(key, message)


def get_messages_from_redis(user_email: str, session_id: str, start: int = 0, end: int = -1):
    key = get_redis_session_key(user_email, session_id)
    messages = redis_conn.lrange(key, start, end)
    return [msg.decode('utf-8') for msg in messages]


class all_messagesResponse(BaseModel):
    messages: List[str]


def scan_keys(user_email: str):
    pattern = f"user:{user_email}:*"
    cursor = '0'  # 초기 cursor는 '0'이어야 합니다.
    keys = []
    while cursor != 0:
        cursor, new_keys = redis_conn.scan(cursor=cursor, match=pattern)
        keys.extend(new_keys)
    messages = []
    for key in keys:
        key = key.decode('utf-8')
        parts = key.split(':')
        session_id = parts[3]
        message = redis_conn.lrange(key, 1, 1)
        if message:
            messages.append(message[0].decode('utf-8'))
    return all_messagesResponse(messages=messages)


def delete_message_from_redis(user_email: str, session_id: str):
    key = f"user:{user_email}:session:{session_id}:messages"
    redis_conn.delete(key)
    return True
