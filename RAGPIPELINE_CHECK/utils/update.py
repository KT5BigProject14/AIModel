# utils.py
import csv
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from tqdm import tqdm

def split_document(doc: Document) -> list[Document]:
    """ LangchainDocument 객체를 받아 적절히 청크로 나눕니다. """
    filename = doc.metadata.get("filename", "")
    if filename.endswith(".csv"):
        return [doc]  # CSV 파일은 이미 로드 시에 청크로 나뉨
    else:
        # 문서 분할(Split Documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(doc)
        return split_documents
        # content = doc.page_content
        # chunk_size = 500
        # chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        # return [Document(page_content=chunk) for chunk in chunks]


def convert_file_to_documents(vector_store, file, SIMILARITY_THRESHOLD):
    """파일을 읽어 Langchain의 Document 객체로 변환"""
    # 유사도 검사를 하지 않고 기존 DB에 넣으면 같은 내용이 계속 쌓임
    
    # 예: file :<_io.BufferedReader name='./documents/sample.txt'>
    
    documents = []
    
    if file.name.endswith(".txt"):
        content = file.read().decode('utf-8')
        results = vector_store.similarity_search_with_score(content, k=1)
        # print(f'유사도 검사 중...results : {results}')
        if results and results[0][1] <= SIMILARITY_THRESHOLD:
            print(f"기존 DB에 유사한 청크가 있음으로 판단되어 추가되지 않음 - {results[0][1]}")
        else: 
            documents = [Document(metadata={"source": file.path, 'page': 0}, page_content=content)]

    elif file.name.endswith(".csv"): 
        loader = CSVLoader(file_path=file.name)
        temp_documents = loader.load()
        for i, row in enumerate(tqdm(temp_documents, total=len(temp_documents), desc='유사도 검사 중...')):
            content = row.page_content
            results = vector_store.similarity_search_with_score(content, k=1)
            # print(f'유사도 검사 중...results : {results}')
            if results and (results[0][1] <= SIMILARITY_THRESHOLD):
                print(f"기존 DB에 유사한 청크가 있음으로 판단되어 추가되지 않음  - {results[0][1]}")
                continue
        
            metadata={"source": row.metadata['source'], 'page': row.metadata['row']}     
            documents.append(Document(metadata=metadata, page_content=content))
            
    elif file.name.endswith(".pdf"):
        loader = PyPDFLoader(file.name)
        temp_documents = loader.load()
        for i, row in enumerate(tqdm(temp_documents, total=len(temp_documents), desc='유사도 검사 중...')):
            content = row.page_content
            results = vector_store.similarity_search_with_score(content, k=1)
            # print(f'유사도 검사 중...results : {results}')
            if results and (results[0][1] <= SIMILARITY_THRESHOLD):
                print(f"기존 DB에 유사한 청크가 있음으로 판단되어 추가되지 않음  - {results[0][1]}")
                continue
            
            metadata={"source": row.metadata['source'], 'page': row.metadata['page']}     
            documents.append(Document(metadata=metadata, page_content=content))
            
    return documents



# def convert_file_to_documents(file):
#     """파일을 읽어 Langchain의 Document 객체로 변환"""
#     # 예: file :<_io.BufferedReader name='./documents/sample.txt'>
#     file_content = file.read().decode('utf-8')
#     if file.name.endswith(".csv"): # 파일이 csv 확장자라면, row 단위로 읽어서 리스트로 변환
#         documents = []
#         reader = csv.reader(file_content.splitlines()) 
#         for i, row in enumerate(reader):
#             content = ",".join(row)
#             metadata = {"filename": file.name, "chunk": i}
#             documents.append(Document(page_content=content, metadata=metadata))
    
#     else: # 나머지 확장자는 전체 파일 내용을 하나의 Document 객체로 변환 -> 기능 수정 필요
#         documents = [Document(page_content=file_content, metadata={"filename": file.name})]
#     return documents

