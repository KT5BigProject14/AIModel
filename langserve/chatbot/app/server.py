from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
import uvicorn
import pickle
from pydantic import BaseModel
from RAGPipeLine import Ragpipeline
from redis_router import router as redis_router  # Redis 라우터 임포트
from langserve import add_routes
from tqdm import tqdm
from langchain_community.vectorstores import Chroma, Document
from langchain_openai import OpenAIEmbeddings

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


app = FastAPI()

# Initialize RAGPipeline
ragpipe = Ragpipeline()


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ChatRequest(BaseModel):
    question: str
    session_id: str
    user_email: str


class TitleRequest(BaseModel):
    request: str


class TitleResponse(BaseModel):
    response: str


class TextRequest(BaseModel):
    title: str


class TextResponse(BaseModel):
    response: str


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.post("/chain/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    try:
        # Set user email and session ID before calling chat_generation
        ragpipe.current_user_email = chat_request.user_email
        ragpipe.current_session_id = chat_request.session_id
        response = ragpipe.chat_generation(chat_request.question)

        return ChatResponse(response=response["answer"], session_id=chat_request.session_id)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chain/generate/title", response_model=TitleResponse)
async def generate_title(title_request: TitleRequest):
    try:
        # print(f"수신된 데이터: {title_request}")

        title = ragpipe.title_generation(title_request.request)
        print(title['answer'])
        return TitleResponse(response=title['answer'])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chain/generate/text", response_model=TextResponse)
def generate_text(text_request: TextRequest):
    try:
        # print(f"수신된 데이터: {text_request}")
        text = ragpipe.text_generation(text_request.title)

        print(text['answer'])
        return TextResponse(response=text['answer'])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/update-vector-db")
async def update_docs(new_data_file: UploadFile = File(...)):
    try:
        # 업로드된 파일을 저장
        new_data_path = "./database/new_data.pkl"
        with open(new_data_path, "wb") as f:
            f.write(new_data_file.file.read())

        # new_data.pkl 파일을 로드
        with open(new_data_path, 'rb') as f:
            new_docs = pickle.load(f)

        # all_docs.pkl 파일을 로드
        all_docs_path = './database/all_docs.pkl'
        if os.path.exists(all_docs_path):
            with open(all_docs_path, 'rb') as f:
                all_docs = pickle.load(f)
        else:
            all_docs = []

        # OpenAI Embeddings 및 Chroma 설정
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        persist_dir = './database'
        vector_store = Chroma(
            persist_directory=persist_dir,  # 있으면 가져오고 없으면 생성
            embedding_function=embeddings
        )

        # Convert tuples to Document objects
        def convert_to_documents(docs):
            document_list = []
            for doc in docs:
                try:
                    # Assuming each doc is a tuple (content, metadata)
                    content, metadata = doc
                    document_list.append(
                        Document(page_content=content, metadata=metadata))
                except ValueError:
                    print(f"Skipping invalid document: {doc}")
            return document_list

        new_docs_converted = convert_to_documents(new_docs)
        all_docs_converted = convert_to_documents(all_docs)

        # Add new documents to vector store and update all_docs
        for doc in tqdm(new_docs_converted):
            print(f"Updating {doc}...")
            vector_store.add_documents([doc])  # Add to vector store
            all_docs_converted.append(doc)     # Add to all_docs

        # 업데이트된 all_docs를 pickle 파일로 저장
        with open(all_docs_path, 'wb') as file:
            pickle.dump(all_docs, file)

        return JSONResponse(content={"message": "all_docs 객체가 성공적으로 저장되었습니다."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete-vector-db")
async def delete_docs(delete_data_file: UploadFile = File(...)):
    try:
        # 업로드된 파일을 저장
        delete_data_path = "./database/delete_data.pkl"
        with open(delete_data_path, "wb") as f:
            f.write(delete_data_file.file.read())

        # delete_data.pkl 파일을 로드
        with open(delete_data_path, 'rb') as f:
            delete_docs = pickle.load(f)

        # all_docs.pkl 파일을 로드
        all_docs_path = './database/all_docs.pkl'
        if os.path.exists(all_docs_path):
            with open(all_docs_path, 'rb') as f:
                all_docs = pickle.load(f)
        else:
            raise HTTPException(
                status_code=404, detail="all_docs.pkl 파일을 찾을 수 없습니다.")

        # OpenAI Embeddings 및 Chroma 설정
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        persist_dir = './database'
        vector_store = Chroma(
            persist_directory=persist_dir,  # 있으면 가져오고 없으면 생성
            embedding_function=embeddings
        )

        # Convert tuples to Document objects
        def convert_to_documents(docs):
            document_list = []
            for doc in docs:
                try:
                    # Assuming each doc is a tuple (content, metadata)
                    content, metadata = doc
                    document_list.append(
                        Document(page_content=content, metadata=metadata))
                except ValueError:
                    print(f"Skipping invalid document: {doc}")
            return document_list

        delete_docs_converted = convert_to_documents(delete_docs)
        all_docs_converted = convert_to_documents(all_docs)

        # Remove documents from vector store and all_docs
        remaining_docs = []
        for doc in tqdm(all_docs_converted):
            if doc not in delete_docs_converted:
                remaining_docs.append(doc)
            else:
                print(f"Deleting {doc} from vector store...")
                vector_store.delete_documents(
                    [doc])  # Remove from vector store

        # 업데이트된 all_docs를 pickle 파일로 저장
        with open(all_docs_path, 'wb') as file:
            pickle.dump(remaining_docs, file)

        return JSONResponse(content={"message": "all_docs 객체가 성공적으로 업데이트되었습니다."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def print_text(self, question: str):

    return question + question


app.include_router(redis_router, prefix="/redis")  # Redis 라우터 추가

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
