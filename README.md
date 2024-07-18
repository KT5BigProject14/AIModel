# 사용법


## terminal

```bash
conda create -n (환경_이름) python=3.11.9

conda activate (환경_이름)

cd langserve/AImodel/chat/

pip install -r requirements.txt
```


#### env. 넣기

#### /langserve/chatbot/app/database 에 chromadb 전체 집어넣기

```bash
cd app

uvicorn server:app --host localhost --port 8080 --reload

```


## swagger

```bash
= chat == # output : answer

"query" : 질문

"session_id": 세션 아이디

"user_email": 유저,

"retrieval_method_mq_sq_pd_en": 리트리벌 방법(mq, sq, pd, basic, 그 외엔 ensemble기반 mq)


= title == # output : title 5개

"query" : 질문


= text == # output : title 기반 text content 3 ~ 5개

"query" : 질문
```


### data team은 RAGPipeLine.py -> def chat_generation() 부분 마지막 print(response) 주석 해제해야 참조 context 확인 가능합니다.
