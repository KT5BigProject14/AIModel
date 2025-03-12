import os
from dotenv import load_dotenv
import redis

load_dotenv()


def get_redis_connection():
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT"))
    REDIS_DATABASE = int(os.getenv("REDIS_DATABASE", 0))

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DATABASE)
        r.ping()
        return r
    except redis.ConnectionError as e:
        print(f"Redis connection error: {e}")
        return None


redis_conn = get_redis_connection()
