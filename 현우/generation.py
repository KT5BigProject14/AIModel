from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

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
    "search_type" : 'mmr',  # similarity
    "similarity_k" : 0.25, # 유사도 측정시 기준 값
    "retriever_k" : 5, # top k 개 청크 가져오기
}

# Agent-Tool 사용 시도해보기
class RAGPipeline:
    def __init__(self, file_path):
        """ 객체 생성 시 멤버 변수 초기화 """
        self.SIMILARITY_THRESHOLD = config["similarity_k"]
        
        
        
        self.llm = ChatOpenAI(
            model       = config['llm_predictor']['model_name'],
            temperature = config['llm_predictor']['temperature']
        )
        
        self.file_path = file_path

        self.vector_store   = self.init_vector_store(self.file_path)
        self.retriever  = self.init_retriever()
        # self.chain      = self.init_chain()
        self.session_histories = {}    
        
        
    def init_vector_store(self, file_path:str):
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        print(f"문서의 페이지수: {len(docs)}")
        
        # 단계 2: 문서 분할(Split Documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)
        print(f"분할된 청크의수: {len(split_documents)}")
        
        """ Vector store 초기화 """
        embeddings = OpenAIEmbeddings( model= config['embed_model']['model_name'] )
        
        
        vector_store = FAISS.from_documents(
            split_documents,
            embeddings
        )
        print(f"[초기화] vector_store 초기화 완료")
        return vector_store
    
    def init_retriever(self):
        """ Retriever 초기화 
        다른 검색방법 사용해보기
        Hybrid Search
        """
        retriever = self.vector_store.as_retriever(
            search_kwargs = {"k": config["retriever_k"]},
            search_type   = config["search_type"] # "similarity"
        )
        print(f"[초기화] retriever 초기화 완료")
        return retriever