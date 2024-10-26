from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List
import shutil
import os
from langchain_utils import generate_data_store, query_database # Import your function

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
)

DATA_PATH = Path("database")
CHROMA_PATH = "ChromaDatabases"
DATA_PATH.mkdir(exist_ok=True)

class ChatRequest(BaseModel):
    collection_name: str
    query: str

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = []

@app.post("/homepage")
async def upload_document(file, user_name):
    # print(file, user_name)
    # # return "meau"
    if not file.filename.endswith('.txt'):
        return JSONResponse(content={"error": "Only .txt files are allowed"}, status_code=400)

    # Define file location
    file_location = DATA_PATH / file.filename

    try:
        if file_location.exists():
            return JSONResponse(content={"error": "File already exists"}, status_code=409)
        
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(file_location)
        generate_data_store(user_name)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    return JSONResponse(content={"message": f"File '{file.filename}' uploaded and processed for user '{user_name}'"})


@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    try:
        response_text, sources = query_database(request.query, request.collection_name)
        
        return ChatResponse(response=response_text, sources=sources or [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete/{user_name}/{file_name}")
async def delete_document(user_name: str, file_name: str):
    # Define file location
    file_location = DATA_PATH / file_name

    try:
        if not file_location.exists():
            return JSONResponse(content={"error": "File not found"}, status_code=404)
        
        os.remove(file_location)  # Remove the file
        return JSONResponse(content={"message": f"File '{file_name}' deleted for user '{user_name}'"})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)