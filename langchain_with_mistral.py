from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

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

template = '''
{chat_history}
"<s>[INST] {question} [/INST]"
'''

prompt = PromptTemplate(input_variables=["chat_history", "question"],template=template)

memory = ConversationBufferMemory(memory_key="chat_history")


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU. - for mixtra 8 x 7b it's around 10 for 16GB GPU
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
n_ctx = 20000
max_tokens = 5000

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="models/7B/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx = n_ctx,
    max_tokens = max_tokens,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
# Chain

llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory)

while True:

    question = getMultilineInput()

    if question == "/bye":
        exit

    response = llm_chain.invoke({"question": question})
   