from llama_cpp import Llama
import json

llm = Llama(model_path="models/30B/mixtral-8x7b-v0.1.Q4_K_M.gguf", chat_format="llama-2",n_gpu_layers=3, n_ctx=2048)

lines = []
while True:
    chunk = input("Enter your input: ")
    if(chunk == "[/INST]"):
        break
    elif(chunk != "[INST]"):
        lines.append(chunk)
            
    
initial_input = "\n".join(lines)

prompt = f"""\
<s> [INST] {initial_input} [/INST] Assistant: </s>""" # Prompt

while True:
    # Get user input

    output = llm(
    prompt,  
    max_tokens=816, # Generate up to 816 tokens
    echo=True # Echo the prompt back in the output
    ) # Generate a completion, can also call create_completion
    
    print(output['choices'][0]['text'])
    print(json.dumps(output["usage"], indent=4))

    user_input = input("Enter your input: ")

    prompt = f"{output['choices'][0]['text']}\n [INST]{user_input}\n [/INST]"

    # Check if user wants to exit
    if user_input == "/bye":
        print("Exiting program...")
        break
    
    



