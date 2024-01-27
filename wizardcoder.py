from llama_cpp import Llama
import json

llm = Llama(model_path="models/30B/wizardcoder-33b-v1.1.Q5_K_M.gguf",n_gpu_layers=6, n_ctx=4000)

lines = []
while True:
    chunk = input("Enter your input: ")
    if(chunk == "[/INST]"):
        break
    elif(chunk != "[INST]"):
        lines.append(chunk)
            
    
initial_input = "\n".join(lines)

prompt = f"""\
[Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{initial_input}\n\n### Response:""" # Prompt

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

    prompt = F"{output['choices'][0]['text']}\n \n\n### Instruction:\n{user_input}\n\n### Response:"

    # Check if user wants to exit
    if user_input == "/bye":
        print("Exiting program...")
        break
    
    



