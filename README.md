사용법

env. 넣기
환경설정(requirements.txt)

== terminal ==


cd langserve/AImodel/chat/app

uvicorn server:app --host 0.0.0.0 --port 8080 --reload


== swagger == 


= chat == # output : answer

"query" : 질문

"session_id": 세션 아이디

"user_email": 유저,

"retrieval_method_mq_sq_pd_en": 리트리벌 방법(mq, sq, pd, basic, 그 외엔 ensemble기반 mq)


= title == # output : title 5개

"query" : 질문


= text == # output : title 기반 text content 3 ~ 5개

"query" : 질문
