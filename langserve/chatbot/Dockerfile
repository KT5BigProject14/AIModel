FROM python:3.11-slim

# 작업 디렉터리 설정
WORKDIR /app

# requirements 파일을 이미지에 복사
COPY requirements.txt .

# pip 업그레이드
RUN pip install --upgrade pip

# 파이썬 종속성 설치
RUN pip install -r requirements.txt

# 나머지 애플리케이션 코드 복사
COPY . .

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
