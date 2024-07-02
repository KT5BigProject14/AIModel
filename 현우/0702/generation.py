'''
인덱스 생성, RAG 파이프라인 구축
'''
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

# utils 폴더의 경로를 추가
# sys.path.append(os.path.abspath(os.path.join('..', 'utils')))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.config import config

from utils.prompt import contextualize_q_prompt, qa_prompt

def main():
    print('hello')

# from .models import Documents


# Agent-Tool 사용 시도해보기
class RAGPipeline:
    
    def __init__(self):
        """ 객체 생성 시 멤버 변수 초기화 """
        self.SIMILARITY_THRESHOLD = config["similarity_k"]
        self.llm = ChatOpenAI(
            model       = config['llm_predictor']['model_name'],
            temperature = config['llm_predictor']['temperature']
        )
        
        self.vector_store   = self.init_vector_store()
        self.retriever      = self.init_retriever()  
        
    
    def init_vector_store(self):
        """ Vector store 초기화 """
        embeddings = OpenAIEmbeddings( model=config['embed_model']['model_name'] )
        vector_store = Chroma(
            persist_directory=config["chroma"]["persist_dir"],  # 있으면 가져오고 없으면 생성
            embedding_function=embeddings
        )
        print(f"[초기화] vector_store 초기화 완료")
        return vector_store
    
    
    def input_pdf(self, file_path):
        loader = PyMuPDFLoader(file_path, extract_images=False)
        docs = loader.load()
        print(f"문서의 페이지수: {len(docs)}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)
        print(f"분할된 청크의수: {len(split_documents)}")
        
        return docs, split_documents
        
    def init_retriever(self):
        """ Retriever 초기화 
        다른 검색방법 사용해보기
        Hybrid Search
        """
        retriever = self.vector_store.as_retriever(
            search_kwargs = {"k": config["retriever_k"]},
            search_type   = "similarity"
        )
        print(f"[초기화] retriever 초기화 완료")
        return retriever


if __name__=="__main__":
    # from .generation import RAGPipeline
    pipeline = RAGPipeline() # 앱 실행 시 전역 파이프라인 객체 생성
    
    vector_store = pipeline.init_vector_store()
    
    print(vector_store.get())
    
    
    file_path = "./raw_data/[법률_규범_특허][코트라][2022]한-인도 CEPA 활용법 및 인도 통상 애로.pdf"
    # print(os.getcwd())
    docs, split_documents = pipeline.input_pdf(file_path)
    # vector_store.add_documents(split_documents)