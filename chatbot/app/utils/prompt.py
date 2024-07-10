from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime

now_time = datetime.today().strftime("%Y/%m/%d %H:%M:%S")

######################################################## 컬럼 생성 프롬프트 #############################################################
# 1. 연관 타이틀 생성 
title_generator_system_prompt ="""
당신은 질문과 질문에 맞는 context를 통해 title을 5개 제시해주는 어시스턴트입니다.
제시된 질문에 연관된 블로그 게시물의 제목, title을 5개 생성해주세요.

---

질문: {input}
맥락 정보: {context}

답변: 
- 
- 
- 
- 
-


예시)
질문: 인도 진출
맥락 정보: 
- 인도는 최근 몇 년간 급속한 경제 성장을 이루고 있습니다.
- 인도 정부는 스타트업 친화적인 정책을 추진하고 있습니다.
- 인도의 주요 산업은 IT, 제조업, 식품 산업입니다.

답변: 
- 인도 진출
- 인도의 최근 경제 동향
- 인도 교역 현황과 경제 전망
- 인도의 식품산업 트렌드
- 인도의 AI 산업 발전 현황

"""

title_generator_prompt = ChatPromptTemplate.from_messages([
    ("system", title_generator_system_prompt),
    ("human", "{input}"),
])


# 2. 게시글 생성 
text_generator_system_prompt="""
당신은 초기 스타트업을 위한 컨텐츠를 작성하는 전문 애널리스트입니다. 
고객의 질문과 주어진 맥락 정보를 바탕으로 신뢰할 수 있고 유용한 정보를 제공하는 게시물을 작성해주세요. 
컨텐츠는 초기 스타트업 창업자들에게 실질적인 도움을 줄 수 있도록 설계되어야 합니다.

고객의 질문:
{input}

제공된 맥락 정보:
{context}

위 정보를 바탕으로 최소 5개 이상의 하위 주제와 단락을 포함한 상세한 게시물을 작성해주세요. 
각 단락은 명확한 제목과 함께 제공되어야 하며, 초보 창업자들에게 이해하기 쉽고 유용해야 합니다.

예시 형식:
## 고객의 질문:
(고객의 질문 내용을 여기에 작성)

## 제공된 맥락 정보:
(제공된 맥락 정보를 여기에 작성)

## 게시물 시작:

### 1. (첫 번째 하위 주제 제목)
(첫 번째 하위 주제 내용)

### 2. (두 번째 하위 주제 제목)
(두 번째 하위 주제 내용)

필요시 추가 하위 주제와 단락을 작성해주세요. 고객의 질문에 대한 포괄적이고 깊이 있는 답변을 제공하는 것이 목표입니다.

"""
text_generator_prompt = ChatPromptTemplate.from_messages([
    ("system", text_generator_system_prompt),
    ("human", "{input}"),
])

# 당신은 주어진 title에 대한 게시물을 작성하는 애널리스트입니다. 
# 관련된 주제에 대해 최신 정보와 주어진 DB정보를 바탕으로 근거있는 게시글을 작성해주세요. 
# 적어도 5개 이상의 수주제와 단락을 생성하여 컬럼 및 게시물을 작성해주세요. 


# """

######################################################## 대화 프롬프트 #############################################################
# 1. 사용자 질문 맥락화 프롬프트
contextualize_q_system_prompt = """
주요 목표는 사용자의 질문을 이해하기 쉽게 다시 작성하는 것입니다.
사용자의 질문과 채팅 기록이 주어졌을 때, 채팅 기록의 맥락을 참조할 수 있습니다.
채팅 기록이 없더라도 이해할 수 있는 독립적인 질문으로 작성하세요.
질문에 바로 대답하지 말고, 필요하다면 질문을 다시 작성하세요. 그렇지 않다면 질문을 그대로 반환합니다.        
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 2. 질문 프롬프트
qa_system_prompt = """
당신은 질문에 대해 정보를 제공해주는 어시스턴트입니다.
소규모 기업 혹은 스타트업이 인도에 수출 사업을 할 때 도움이 되는 사실 기반의 정보를 제공하도록 설계되었습니다.
주어진 자료와 정확한 수치를 참고하여 정보를 제공해주세요. 

제공된 자료:
{context}

고객의 질문:
{input}

답변:


"""


qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Web search 질문 프롬프트

web_qa_system_prompt = """
당신은 질문에 대해 정보를 제공해주는 어시스턴트입니다.
한국 Google에서 검색해서 정보를 알려주세요.
한국어로 대답하세요. 

자료:
{context}

질문:
{input}

"""

web_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", web_qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])