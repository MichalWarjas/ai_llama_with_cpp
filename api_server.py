from fastapi import FastAPI, HTTPException
from typing import List

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    


@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI model interface"}

@app.post("/generate")
async def generate(user_input: str):
    """Endpoint for generating responses"""
    prompt = f"""\
    "
    {user_input}
    """ # Prompt