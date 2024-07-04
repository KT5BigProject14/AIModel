# utils.py
import csv
from langchain.schema import HumanMessage, SystemMessage, Document


def split_document(doc: Document) -> list[Document]:
    """ LangchainDocument 객체를 받아 적절히 청크로 나눕니다. """
    filename = doc.metadata.get("filename", "")
    if filename.endswith(".csv"):
        return [doc]  # CSV 파일은 이미 로드 시에 청크로 나뉨
    else:
        content = doc.page_content
        chunk_size = 500
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        return [Document(page_content=chunk) for chunk in chunks]


def convert_file_to_documents(file):
    """파일을 읽어 Langchain의 Document 객체로 변환"""
    
    file_content = file.read().decode('utf-8')
    if file.name.endswith(".csv"): # 파일이 csv 확장자라면, row 단위로 읽어서 리스트로 변환
        documents = []
        reader = csv.reader(file_content.splitlines()) 
        for i, row in enumerate(reader):
            content = ",".join(row)
            metadata = {"filename": file.name, "chunk": i}
            documents.append(Document(page_content=content, metadata=metadata))
    else: # 나머지 확장자는 전체 파일 내용을 하나의 Document 객체로 변환 -> 기능 수정 필요
        documents = [Document(page_content=file_content, metadata={"filename": file.name})]
    return documents

