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


class Input(BaseModel):
    input: str
    session_id: str
    user_email: str
    
    
class ChatRequest(BaseModel):
    question: str
    session_id: str
    user_email: str
    
class ChatResponse(BaseModel):
    response: str
    session_id: str    

class TitleRequest(BaseModel):
    response: str


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest):
    try:
        # Set user email and session ID before calling chat_generation
        ragpipe.current_user_email = chat_request.user_email
        ragpipe.current_session_id = chat_request.session_id
        response = ragpipe.chat_generation(chat_request.question)
        
        return JSONResponse(content={"response": response["answer"], "session_id": chat_request.session_id})
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missiong field: {e}")
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

@app.post("/chain/stream_log")
async def stream_log(request: Request):
    try:
        body = await request.json()
        input_data = body['input']
        user_email = input_data['user_email']
        session_id = input_data.get('session_id')
        question = input_data.get('input')
        response = ragpipe.chat_generation(question=question)
        response = ragpipe.invoke(
            {"input": question, "session_id": session_id, "user_email": user_email})
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    
# @app.post("/chain/generate/title")
# async def generate_title(request: Request):
# # async def generate_title(request: str):
#     # try:
#         body = await request.json()
#         print(body)
#         print(body['question'])
#         response = ragpipe.title_generation(question=body['question'])
#         print(response)
#         print(response.type)
#         print(JSONResponse(content=body['question']))
#         return JSONResponse(content=body['question'])
#     # except Exception as e:
#     #     raise HTTPException(status_code=422, detail=str(e))
    
def print_text(self, question: str):
        
    return question + question
    
@app.post("/chain/generate/title")
def generate_title(request: str):
    try:
        print(request)
        title = ragpipe.title_generation(request)
        print(title)
        return title
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    
@app.post("/chain/generate/text")
def generate_text(request: str):
    try:
        print(request)
        text = ragpipe.text_generation(request)
        print(text)
        return text
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    
# @app.post("/chain/generate/text")
# async def generate_text(request: Request):
#     try:
#         body = await request.json()
#         input_data = body['input']
#         user_email = input_data['user_email']
#         session_id = input_data.get('session_id')
#         question = input_data.get('input')
        
#         # response = ragpipe.invoke(
#         #     {"input": question, "session_id": session_id, "user_email": user_email})
#         return JSONResponse(content=response)
#     except Exception as e:
#         raise HTTPException(status_code=422, detail=str(e))

app.include_router(redis_router, prefix="/redis")  # Redis 라우터 추가


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
