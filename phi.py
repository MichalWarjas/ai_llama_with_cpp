from llama_cpp import Llama
import json
from fastapi import FastAPI, HTTPException
from typing import List
import sys

llm = None # Initialize the model once, outside of the function scope
local_model_path = "models/4B/Phi-3-mini-4k-instruct-fp16.gguf"
gpu_layers = -1
conversation_history = ""
    
def getMultilineInput():
    lines = []
    while True:
        chunk = input("Enter your input: ")
        if chunk.lower() == '/done':
            break
        elif chunk.lower() == '/bye':
            return chunk.lower()
            
        lines.append(chunk)
                
    joined_input = "\n".join(lines)
    return joined_input

def runPhi(apiMode=False):
    global conversation_history
    global llm
    global local_model_path
    global gpu_layers

    if "8x7" in local_model_path:
        gpu_layers = 12

    llm = Llama(model_path=local_model_path,n_gpu_layers=gpu_layers, n_ctx=10000, n_threads=4)

    print("-----------------------------------------------------------------------")
    print("")
    print(f"Using model:  {local_model_path} GPU layers: {gpu_layers}")
    print("")
    print("-----------------------------------------------------------------------")

    if apiMode:
        print("Running model in API mode")
    else: 
        initial_input = getMultilineInput()

        prompt = f"<|system|>You are a helpful AI assistant.<|end|> <|user|>{initial_input}<|end|><|assistant|>"



        while True:
            # Get user input

            output = llm(
            prompt,  
            max_tokens=816, # Generate up to 816 tokens
            stop=["<|end|>"], 
            echo=True # Echo the prompt back in the output
            ) # Generate a completion, can also call create_completion
            conversation_history = output['choices'][0]['text']
            print(conversation_history)
            print(json.dumps(output["usage"], indent=4))

            user_input = getMultilineInput()

            prompt = F"{conversation_history}<|end|>\n <|user|>\n{user_input}<|end|>\n<|assistant|>"

            # Check if user wants to exit
            if user_input == "/bye":
                print("Exiting program...")
                break

if __name__ == "__main__":
    runPhi(False)


