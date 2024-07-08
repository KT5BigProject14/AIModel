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


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.post("/chat")
async def chat(question: str, session_id: str, user_email: str):
    try:
        # Set user email and session ID before calling chat_generation
        ragpipe.current_user_email = user_email
        ragpipe.current_session_id = session_id
        response = ragpipe.chat_generation(question)
        return {"response": response}
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
        # response = ragpipe.invoke(
        #     {"input": question, "session_id": session_id, "user_email": user_email})
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

app.include_router(redis_router, prefix="/redis")  # Redis 라우터 추가

add_routes(
    app,
    ragpipe.with_types(input_type=Input),
    path="/chain",
    playground_type="default",  # default, chat
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)