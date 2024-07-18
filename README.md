## 사용법


### == terminal ==

1. conda create -n (환경_이름) python=3.11.9

2. conda activate (환경_이름)

3. cd langserve/AImodel/chat/

4. ''' pip install -r requirements.txt '''

5. env. 넣기

6. /langserve/chatbot/app/database 에 chromadb 전체 집어넣기

7. cd app

8. uvicorn server:app --host 0.0.0.0 --port 8080 --reload

9. url : (http://localhost:8080/docs)


### == swagger == 


= chat == # output : answer

"query" : 질문

"session_id": 세션 아이디

"user_email": 유저,

"retrieval_method_mq_sq_pd_en": 리트리벌 방법(mq, sq, pd, basic, 그 외엔 ensemble기반 mq)


= title == # output : title 5개

"query" : 질문


= text == # output : title 기반 text content 3 ~ 5개

"query" : 질문



data team은 RAGPipeLine.py -> def chat_generation() 부분 마지막 print(response) 주석 해제해야 참조 context 확인 가능합니다.
