API developed with Python's FastAPI framework, which enables interaction with a language model using the `llama-cpp` library. The code initializes and runs the Llama language model based on user input or API requests. It supports multiple models such as Dolphin, Phi, Mistral, Mixtral, Bielik, and Codestral.

## Dependencies
- FastAPI
- llama_cpp
- uvicorn (for running the API)
- json
- sys

## Functions
The code includes several functions to handle model initialization, prompt generation, interaction with the language model, and API routing:
1. `isModelInitialized`: Checks if the model is initialized.
2. `getInitializedModel`: Returns the path of the initialized model.
3. `initializeModel`: Initializes the language model by setting `llm` and `model_initialized` global variables to their default values.
4. `runLLM`: Initializes and runs the language model based on input parameters and API mode. It calls the `getAnswer` function in a loop for interactive mode or returns a success message for API mode.
5. `getAnswer`: Generates an answer from the language model based on user input and conversation history using the appropriate prompt format.
6. `getInitialPrompt`: Creates an initial prompt with system instructions and user input. The prompt format varies based on the selected model.
7. `getFollowingPrompt`: Creates a subsequent prompt in a conversation by appending new user input to the existing conversation history.
8. API endpoints (using FastAPI) for running the language model, checking the model's initialization status, and retrieving initialized models.


Llama cpp instalation:

Install llama-cpp-python

https://github.com/abetlen/llama-cpp-python



CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

If gpu offload stops to work use:
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

make sure your cuda toolkit version (nvcc) mathes your driver version (nvidia-smi).
Install both as described here 
https://developer.nvidia.com/cuda-downloads


To run API server run:
uvicorn api_server:app --reload

API
curl -X POST "http://127.0.0.1:8000/generate" -H "accept: application/json" -H "Content-Type: application/json" -d '{
"user_input": "who are you?"}' - generate call is asuynchronous. You need call /answer to check if it is available