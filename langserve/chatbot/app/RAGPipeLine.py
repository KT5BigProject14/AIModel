# RAGPipeLine.py
from fastapi import FastAPI, HTTPException, Request
import logging
from utils.update import split_document, convert_file_to_documents
from utils.prompt import contextualize_q_prompt, qa_prompt
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables import Runnable
from langchain.chains import LLMChain
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.llms import OpenAI
from utils.prompt import *
from utils.config import *
from utils.redis_utils import save_message_to_redis, get_messages_from_redis
from core.redis_config import redis_conn  # Redis 설정 임포트
from langchain_core.runnables import RunnableParallel
# from langchain.retrievers import WebResearchRetriever
from langchain_community.retrievers.web_research import WebResearchRetriever
# from langchain.utilities import GoogleSearchAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
# Retriever 기법
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
# Ensemble retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import pickle

load_dotenv()


class Ragpipeline:
    def __init__(self):
        self.SIMILARITY_THRESHOLD = config["similarity_k"]
        self.llm = ChatOpenAI(
            model=config['llm_predictor']['model_name'],
            temperature=config['llm_predictor']['temperature']
        )
        self.vector_store = self.init_vectorDB()
        self.retriever = self.init_retriever()
        # self.web_retriever = self.init_web_research_retriever()
        self.mq_retriever = self.init_multi_query_retriever()
        self.bm25_retriever = self.init_bm25_retriever()
        self.ensemble_retriever = self.init_ensemble_retriever()
        self.mq_ensemble_retriever = self.init_mq_ensemble_retriever()
        # self.web_chain = self.init_web_chat_chain()
        self.title_chain = self.init_title_chain()
        self.text_chain = self.init_text_chain()
        self.ensemble_chain = self.init_ensemble_chain()
        self.mq_ensemble_chain = self.init_mq_ensemble_chain()
        self.session_histories = {}
        self.current_user_email = None
        self.current_session_id = None

    def init_vectorDB(self):
        embeddings = OpenAIEmbeddings(
            model=config['embed_model']['model_name'])
        vector_store = Chroma(
            persist_directory=config["chroma"]["persist_dir"],
            embedding_function=embeddings
        )
        return vector_store

    def init_retriever(self):
        # retriever = self.vector_store.as_retriever(
        #         search_kwargs = {"score_threshold": 0.4, "k": 5},
        #         search_type   = "similarity_score_threshold"
        #     )
        retriever = self.vector_store.as_retriever(
            search_kwargs={'fetch_k': 10, "k": 5, 'lambda_mult': 0.4},
            search_type="mmr"
        )

        return retriever

    def init_bm25_retriever(self):

        with open(config["pkl_path"], 'rb') as file:
            all_docs = pickle.load(file)

        bm25_retriever = BM25Retriever.from_documents(
            all_docs
        )
        bm25_retriever.k = 5

        return bm25_retriever

    def init_ensemble_retriever(self):

        bm25_retriever = self.bm25_retriever
        chroma_retriever = self.retriever

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.4, 0.6],
            search_type=config["ensemble_search_type"],
        )

        return ensemble_retriever

    def init_mq_ensemble_retriever(self):

        ensemble_retriever = self.ensemble_retriever

        mq_ensemble_retriever = MultiQueryRetriever.from_llm(
            retriever=ensemble_retriever, llm=self.llm
        )

        return mq_ensemble_retriever

    # def init_web_research_retriever(self):
    #     """ Web Research Retriever 초기화 """
    #     search = GoogleSearchAPIWrapper()
    #     allow_dangerous_requests = False
    #     vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),
    #                          persist_directory="./temp_web_db")
    #     llm_chain = LLMChain(
    #     llm=self.llm
    # )
    #     web_retriever = WebResearchRetriever(vectorstore = vectorstore, llm_chain = llm_chain, search = search, allow_dangerous_requests = False).from_llm(
    #         vectorstore=vectorstore,
    #         llm=self.llm,
    #         search=search,
    #     )
    #     # web_retriever = web_retriever.from_llm(
    #     #     vectorstore=vectorstore,
    #     #     llm=self.llm,
    #     #     search=search,
    #     # )
    #     return web_retriever

    def init_multi_query_retriever(self):
        """사용자의 질문을 여러 개의 유사 질문으로 재생성 """
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=self.retriever, llm=self.llm
        )

        return retriever_from_llm

    def init_web_chat_chain(self):
        # 1. 사용자의 질문 문맥화 <- 프롬프트 엔지니어링
        history_aware_retriever = create_history_aware_retriever(                           # 대화 기록을 가져온 다음 이를 사용하여 검색 쿼리를 생성하고 이를 기본 리트리버에 전달
            self.llm, self.web_retriever, contextualize_q_prompt
        )
        # 2. 응답 생성 + 프롬프트 엔지니어링
        # 문서 목록을 가져와서 모두 프롬프트로 포맷한 다음 해당 프롬프트를 LLM에 전달합니다.
        question_answer_chain = create_stuff_documents_chain(
            self.llm, web_qa_prompt)

        # 3. 최종 체인 생성
        # 사용자 문의를 받아 리트리버로 전달하여 관련 문서를 가져옵니다. 그런 다음 해당 문서(및 원본 입력)는 LLM으로 전달되어 응답을 생성
        rag_chat_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)

        return rag_chat_chain
    
    def init_ensemble_chain(self):
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.ensemble_retriever, contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(
            self.llm, qa_prompt)
        rag_chat_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)
        print("[갱신] ensemble RAG chain history 갱신 완료")
        return rag_chat_chain
    

    def init_mq_ensemble_chain(self):

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.mq_ensemble_retriever, contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(
            self.llm, qa_prompt)
        rag_chat_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)
        return rag_chat_chain

    def init_title_chain(self):
        question_answer_chain = create_stuff_documents_chain(
            self.llm, title_generator_prompt)
        rag_title_chain = create_retrieval_chain(
            self.retriever, question_answer_chain)
        return rag_title_chain

    def init_text_chain(self):
        question_answer_chain = create_stuff_documents_chain(
            self.llm, text_generator_prompt)
        rag_text_chain = create_retrieval_chain(
            self.mq_ensemble_retriever, question_answer_chain)
        return rag_text_chain

    def chat_generation(self, question: str) -> dict:
        def get_session_history(session_id=None, user_email=None):
            session_id = session_id if session_id else self.current_session_id
            user_email = user_email if user_email else self.current_user_email

            if session_id not in self.session_histories:
                self.session_histories[session_id] = ChatMessageHistory()
                # Redis에서 세션 히스토리 불러오기
                history_messages = get_messages_from_redis(
                    user_email, session_id)
                for message in history_messages:
                    self.session_histories[session_id].add_message(
                        HumanMessage(content=message)
                    )
            return self.session_histories[session_id]

        results = self.vector_store.similarity_search_with_score(question, k=1)

        # if results[0][1] > 0.3:  # web chain
        #     final_chain = self.web_chain
        # else:
        final_chain = self.ensemble_chain


        conversational_rag_chain = RunnableWithMessageHistory(
            final_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        response = conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": self.current_session_id}}
        )

        # Redis에 세션 히스토리 저장
        save_message_to_redis(self.current_user_email,
                              self.current_session_id, question)
        save_message_to_redis(self.current_user_email,
                              self.current_session_id, response["answer"])
        print(response)
        return response

    def title_generation(self, question: str):
        title_chain = self.title_chain
        response = title_chain.invoke({'input': question})
        print(response)
        return response

    def text_generation(self, question: str):

        text_chain = self.text_chain
        response = text_chain.invoke({'input': question})
        print(response)
        return response
