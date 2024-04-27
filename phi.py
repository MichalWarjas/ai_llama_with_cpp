from llama_cpp import Llama
import json

local_model_path = "models/4B/Phi-3-mini-4k-instruct-fp16.gguf"
gpu_layers = -1
multiline_input = False

if "8x7" in local_model_path:
    gpu_layers = 12

llm = Llama(model_path=local_model_path,n_gpu_layers=gpu_layers, n_ctx=10000, n_threads=4)

print("-----------------------------------------------------------------------")
print("")
print(f"Using model:  {local_model_path} GPU layers: {gpu_layers}")
print("")
print("-----------------------------------------------------------------------")

if multiline_input:
    lines = []

    while True:
        chunk = input("Enter your input: ")
        if(chunk == "[/INST]"):
            break
        elif(chunk != "[INST]"):
            lines.append(chunk)
                
    initial_input = "\n".join(lines)
else:
    initial_input = input("Enter your input: ")

prompt = f"""\
"<|user|>\n{initial_input}<|end|>\n<|assistant|>"
""" # Prompt



while True:
    # Get user input

    output = llm(
    prompt,  
    max_tokens=816, # Generate up to 816 tokens
    stop=["<|end|>"], 
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
    
    



