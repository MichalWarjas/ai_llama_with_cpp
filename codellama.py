from llama_cpp import Llama
llm = Llama(model_path="models/30B/Wizard-Vicuna-30B-Uncensored.Q5_K_M.gguf", chat_format="llama-2",n_gpu_layers=6)
output = llm(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion

print(output)