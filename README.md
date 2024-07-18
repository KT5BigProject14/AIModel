사용법


== terminal ==
환경설정
conda create -n (환경_이름) python=3.11.9

conda activate (환경_이름)

cd langserve/AImodel/chat/

pip install -r requirements.txt
(requirements.txt)

env. 넣기

cd app

uvicorn server:app --host 0.0.0.0 --port 8080 --reload

url : (http://localhost:8080/docs)


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
