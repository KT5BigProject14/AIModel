# redis_utils.py
from core.redis_config import redis_conn


def get_redis_session_key(user_email: str, session_id: str) -> str:
    return f"user:{user_email}:session:{session_id}:messages"


def save_message_to_redis(user_email: str, session_id: str, message: str):
    key = get_redis_session_key(user_email, session_id)
    redis_conn.lpush(key, message)


def get_messages_from_redis(user_email: str, session_id: str, start: int = 0, end: int = -1):
    key = get_redis_session_key(user_email, session_id)
    messages = redis_conn.lrange(key, start, end)
    return [msg.decode('utf-8') for msg in messages]


def scan_keys(user_email: str):
    pattern = f"user:{user_email}:*"
    cursor = 0
    keys = []
    while cursor != 0:
        cursor, new_keys = redis_conn.scan(cursor=cursor, match=pattern)
        keys.extend(new_keys)
    messages = []
    for key in keys:
        key = key.decode('utf-8')
        parts = key.split(':')
        session_id = parts[3]
        message = redis_conn.lrange(key, -2, -2)
        if message:
            messages.append({session_id: message[0].decode('utf-8')})
    return messages


def delete_message_from_redis(user_email: str, session_id: str):
    key = f"user:{user_email}:session:{session_id}:messages"
    redis_conn.delete(key)
    return True
