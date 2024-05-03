from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import llama3_chat

class Query(BaseModel):
    user_input: str

def start_llm():
    
    model_path = "models/4B/Phi-3-mini-4k-instruct-fp16.gguf"
    system_message = "You are a helpful AI assistant."

    llama3_chat.runLLM(model_path, system_message, True)

start_llm()

app = FastAPI()

origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
   print('startup event triggered') 

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI model interface"}

@app.post("/generate")
async def generate(user_input: Query):
    print(f"question received: {user_input.user_input}")
    response = llama3_chat.getAnswer(user_input.user_input)
    split_string = response.split('<|assistant|>')
    final_answer = split_string[-1].strip()
    return {"generated_response": final_answer}