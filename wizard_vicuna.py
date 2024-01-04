from llama_cpp import Llama
llm = Llama(model_path="models/30B/Wizard-Vicuna-30B-Uncensored.Q5_K_M.gguf", chat_format="llama-2",n_gpu_layers=6)

initial_input = input("Enter your input: ")

prompt = f"""\
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
USER: {initial_input}
ASSISTANT:""" # Prompt

while True:
    # Get user input

    output = llm(
    prompt,  
    max_tokens=2048, # Generate up to 32 tokens
    echo=True # Echo the prompt back in the output
    ) # Generate a completion, can also call create_completion
    
    print(output)

    user_input = input("Enter your input: ")

    prompt = F"{output['choices'][0]['text']}\n USER: {user_input}\n ASSISTANT:"

    # Check if user wants to exit
    if user_input == "/bye":
        print("Exiting program...")
        break
    
    



