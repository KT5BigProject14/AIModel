import csv
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables import RunnableBranch, RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain.agents import Tool, create_openai_tools_agent, AgentExecutor
# from langchain import hub
# from langchain.tools.retriever import create_retriever_tool
# from llama_index.core import (
#     SimpleDirectoryReader
# )

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import sys

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

from utils.prompt import contextualize_q_prompt, qa_prompt
from utils.update import split_document, convert_file_to_documents

# intfloat/multilingual-e5-small
config = {
    "llm_predictor" : {
        "model_name"  : "gpt-3.5-turbo", # gpt-3.5-turbo-0613,
        "temperature" : 0
    },
    "embed_model" : {
        "model_name" : "text-embedding-ada-002", # "intfloat/e5-small",
        "cache_directory" : "",
    },
    "chroma" : {
        "persist_dir" : "./database",
    },
    "path" : {
        "input_directory" : "./documents",
    },
    "search_type" : "similarity",   # "mmr"
    "ensemble_search_type" : "mmr",
    "similarity_k" : 0.25, # 유사도 측정시 기준 값
    "retriever_k" : 5, # top k 개 청크 가져오기
}

'''
# RAGpipeline
1. 파이프라인 생성시 
    1. 초기 vectorDB 생성 
    2. 검색기 생성 
    
2. pdf 입력시
    1. pdf를 잘라서 vectorDB에 넣기 

3. 사용자가 검색할 때,
    1. 검색에 대한 연관 title 생성 
    2. 생성된 title에 대한 게시글 생성 

4. 사용자가 채팅할 때, 
    1. 채팅 세션 생성 
    2. 채팅 

'''

class Ragpipeline:
    
    def __init__(self):
        """ 객체 생성 시 멤버 변수 초기화 """
        self.SIMILARITY_THRESHOLD = config["similarity_k"]
        
        self.llm = ChatOpenAI(
            model       = config['llm_predictor']['model_name'],
            temperature = config['llm_predictor']['temperature']
        )
        
        self.vector_store   = self.init_vectorDB()
        self.retriever      = self.init_retriever()  
        
    def init_vectorDB(self):
        """ Vector store 초기화 """
        embeddings = OpenAIEmbeddings( model=config['embed_model']['model_name'] )                      # 나중에 허깅페이스에서 임베딩 모델 가져오기 
        vector_store = Chroma(
            persist_directory=config["chroma"]["persist_dir"],  # 있으면 가져오고 없으면 생성
            embedding_function=embeddings
        )
        print(f"[초기화] vector_store 초기화 완료")
        return vector_store
    
    def init_retriever(self):
        """ Retriever 초기화 """                        # 나중에 FAISS랑 BM25와 함께 Hybrid 또는 EnsembleRetriever 적용 
        retriever = self.vector_store.as_retriever(
            search_kwargs = {"k": config["retriever_k"]},
            search_type   = "similarity"
        )
        print(f"[초기화] retriever 초기화 완료")
        return retriever
    
    def init_chat_chain(self):
        """ chain 초기화 
        리트리버 전용 체인으로 변경해보기
        create_history_aware_retriever : 대화 기록을 가져온 다음 이를 사용하여 검색 쿼리를 생성하고 이를 기본 리트리버에 전달
        """
        
        # 사용자의 질문 문맥화 <- 프롬프트 엔지니어링
        history_aware_retriever = create_history_aware_retriever(                           # 대화 기록을 가져온 다음 이를 사용하여 검색 쿼리를 생성하고 이를 기본 리트리버에 전달
            self.llm, self.retriever, contextualize_q_prompt
        )
        
        # 응답 생성 + 프롬프트 엔지니어링
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)           # 문서 목록을 가져와서 모두 프롬프트로 포맷한 다음 해당 프롬프트를 LLM에 전달합니다.
        
        # 최종 체인 생성
        rag_chat_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)  # 사용자 문의를 받아 리트리버로 전달하여 관련 문서를 가져옵니다. 그런 다음 해당 문서(및 원본 입력)는 LLM으로 전달되어 응답을 생성

        print("[초기화] RAG chain 초기화 완료")
        return rag_chat_chain
    
    def chat_generation(self, question: str, session_id: str) -> dict:
        # 채팅 기록 관리
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.session_histories:
                self.session_histories[session_id] = ChatMessageHistory()
                print(f"[히스토리 생성] 새로운 히스토리를 생성합니다. 세션 ID: {session_id}")
            return self.session_histories[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            self.chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        response = conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )

        print(f'[응답 생성] 실제 모델 응답: response => \n{response}\n')
        print(f"[응답 생성] 세션 ID [{session_id}]에서 답변을 생성했습니다.")

        # Redis에 대화 기록 저장
        chat_history_key = f"chat_history:{session_id}"
        try:
            print(f'--------> Redis 키: {chat_history_key}')
            chat_history = self.redis_client.get(chat_history_key)
            print('-------->',chat_history)
            if chat_history:
                chat_history = json.loads(chat_history)
            else:
                chat_history = []

            chat_history.append({"question": question, "answer": response["answer"]})
            print(f'=========> 저장할 대화 기록: {chat_history}')
            print(f'=========> 저장할 대화 기록 json: {json.dumps(chat_history)}')
            self.redis_client.set(chat_history_key, json.dumps(chat_history))        
            print(f"[Redis 저장] 세션 ID [{session_id}]의 대화 기록을 Redis에 저장했습니다.")
        except Exception as e:
            print(f"[Redis 저장 실패] 세션 ID [{session_id}]의 대화 기록 저장 중 오류 발생: {e}")

        return response["answer"]
    
    def update_vector_db(self, file, filename) -> bool:
        """
        벡터 스토어 업데이트: 새로운 문서 추가 시 호출 
        PDF파일 또는 CSV파일 또는 hwp파일, word 파일 등 
        기존 DB에 유사도 검사를 통해 중복되는 내용은 추가하지 않음 
        """
        upload_documents = convert_file_to_documents(self.vector_store, file, self.SIMILARITY_THRESHOLD)

        if upload_documents:
            self.vector_store.add_documents(upload_documents)
            print(f"Added {len(upload_documents)} new documents to the vector store")
            return True
        else:
            print('모두 유사한 청크로 판단되어 해당 문서가 저장되지 않음')
            return False
    

    def delete_vector_db_by_doc_id(self, doc_id):
        """
        주어진 문서 ID에 해당하는 벡터 임베딩을 삭제
        """
        # 벡터 데이터베이스에서 모든 문서 가져오기
        all_documents = self.vector_store._collection.get(include=["metadatas"])
        documents_to_delete = [doc_id for i, metadata in enumerate(all_documents["metadatas"]) if metadata.get("doc_id") == doc_id]
        if documents_to_delete:
            self.vector_store._collection.delete(ids=documents_to_delete)
            print(f"[벡터 DB 삭제] 문서 ID [{doc_id}]의 임베딩을 벡터 DB에서 삭제했습니다.")
        else:
            print(f"[벡터 DB 삭제 실패] 문서 ID [{doc_id}]에 대한 임베딩을 찾을 수 없습니다.")
    
        
if __name__=="__main__":
    ragpipe = Ragpipeline()
    # vectorDB = ragpipe.init_vectorDB()
    