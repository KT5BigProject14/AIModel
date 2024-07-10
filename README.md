## 시작 

```bash
pip install -r --ignore-installed requirements.txt
```

## Redis 연결 

Redis 연결 하지 않으면 다음과 같은 에러가 뜬다.

Redis connection error: Error 10061 connecting to localhost:6379. 대상 컴퓨터에서 연결을 거부했으므로 연결하지 못했습니다.

**Window용 Resis설치**

https://github.com/microsoftarchive/redis/releases

Redis-x64-3.0.504.msi

## 데이터베이스 

아래 링크 다운받고 app 폴더에 넣기 

https://drive.google.com/file/d/1WXQ5yf8XPIXURwq1cFmmrEpBYYiPOzBc/view?usp=drive_link

## 실행 

```bash
cd chatbot/app
```

```bash
python server.py
```
