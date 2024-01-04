from llama_cpp import Llama
llm = Llama(model_path="models/30B/codellama-34b-instruct.Q5_K_M.gguf", chat_format="llama-2",n_gpu_layers=6, n_ctx=2048)

initial_input = input("Enter your input: ")

prompt = f"""\
[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases.
 Please wrap your code answer using ```:
 {initial_input}
[/INST]""" # Prompt

while True:
    # Get user input

    output = llm(
    prompt,  
    max_tokens=816, # Generate up to 816 tokens
    echo=True # Echo the prompt back in the output
    ) # Generate a completion, can also call create_completion
    
    print(output)

    user_input = input("Enter your input: ")

    prompt = F"{output['choices'][0]['text']}\n [INST]{user_input}\n [/INST]"

    # Check if user wants to exit
    if user_input == "/bye":
        print("Exiting program...")
        break
    
    



