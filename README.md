API for various locals language models


Requirements:
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
"user_input": "who are you?"}'