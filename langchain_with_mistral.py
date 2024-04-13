from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

template = '''
"<s>[INST] {question} [/INST]"
'''

prompt = PromptTemplate.from_template(template)



# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU. - for mixtra 8 x 7b it's around 10 for 16GB GPU
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
n_ctx = 20000
max_tokens = 5000

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="models/7B/bielik-7b-instruct-v0.1.Q8_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx = n_ctx,
    max_tokens = max_tokens,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
# Chain

while True:

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = input("Enter your input: ")

    response = llm_chain.invoke({"question": question})

    r_question = response['question']
    r_text = response['text']

    r_question_placeholder = "{question}"
    following_template = f'''
    "<s>[INST] {r_question} [/INST]\n
    {r_text}\n\n
    [INST] {r_question_placeholder} [/INST]
    "  
    '''

    prompt = PromptTemplate.from_template(following_template)


# llm.invoke(question)
