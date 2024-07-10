from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
import uvicorn
from pydantic import BaseModel
from RAGPipeLine import Ragpipeline
from redis_router import router as redis_router  # Redis 라우터 임포트
from langserve import add_routes

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

        return ChatResponse(response=response, session_id=chat_request.session_id)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chain/generate/title", response_model=TitleResponse)
async def generate_title(title_request: TitleRequest):
    try:
        # print(f"수신된 데이터: {title_request}")

        title = ragpipe.title_generation(title_request.request)
        # print(f"생성된 제목: {title['answer']}")

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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-vector-db")
async def update_vector_db(file: UploadFile = File(...)):
    try:
        filename = file.filename
        file_content = await file.read()
        success = ragpipe.update_vector_db(file_content, filename)
        if success:
            return {"status": "success", "message": "Vector store updated successfully."}
        else:
            return {"status": "failed", "message": "Document was too similar to existing entries."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-vector-db")
async def delete_vector_db(doc_id: str):
    try:
        ragpipe.delete_vector_db_by_doc_id(doc_id)
        return {"status": "success", "message": f"Document with ID {doc_id} deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def print_text(self, question: str):

    return question + question


app.include_router(redis_router, prefix="/redis")  # Redis 라우터 추가

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
