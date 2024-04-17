from llama_cpp import Llama
import json
import sys

if __name__ == "__main__":

    default_model = "models/7B/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    model_name = ""
    gpu_layers = -1

    if len(sys.argv) != 2:
        print("parameters usage: python(3) mixtral.py model=(7/8x7/dolphin)")
    else:
        param_splitted = sys.argv[1].split("=")
        if(len(param_splitted) != 2):
            print("parameters usage: python(3) mixtral.py model=(7/8x7/dolphin)")
        else:
            if param_splitted[1] == "8x7":
                model_name = "models/mistral/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"
                gpu_layers = 12
            elif param_splitted[1] == "dolphin":
                model_name = "models/mistral/dolphin-2.7-mixtral-8x7b.Q5_K_M.gguf"
                gpu_layers = 12

    mistral_model = default_model if model_name == "" else model_name

    llm = Llama(model_path=mistral_model,n_gpu_layers=gpu_layers, n_ctx=4000, n_threads=4)

    lines = []
    print("-----------------------------------------------------------------------")
    print("")
    print(f"Using model:  {mistral_model} GPU layers: {gpu_layers}")
    print("")
    print("-----------------------------------------------------------------------")

    while True:
        chunk = input("Enter your input: ")
        if(chunk == "[/INST]"):
            break
        elif(chunk != "[INST]"):
            lines.append(chunk)
                
        
    initial_input = "\n".join(lines)

    prompt = f"""\
    [INST]{initial_input}[/INST]
    """ # Prompt

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
    
    



