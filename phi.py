from llama_cpp import Llama
import json

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

local_model_path = "models/4B/Phi-3-mini-4k-instruct-fp16.gguf"
gpu_layers = -1

if "8x7" in local_model_path:
    gpu_layers = 12

llm = Llama(model_path=local_model_path,n_gpu_layers=gpu_layers, n_ctx=10000, n_threads=4)

print("-----------------------------------------------------------------------")
print("")
print(f"Using model:  {local_model_path} GPU layers: {gpu_layers}")
print("")
print("-----------------------------------------------------------------------")

initial_input = getMultilineInput()

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

    user_input = getMultilineInput()

    prompt = F"{output['choices'][0]['text']}\n \n\n### Instruction:\n{user_input}\n\n### Response:"

    # Check if user wants to exit
    if user_input == "/bye":
        print("Exiting program...")
        break
    
    



