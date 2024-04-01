from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate.from_template(template)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
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

question = """
Is there a way to check how my entry gate remote control can be set to copy mode? I cant find any info about the model
"""
llm.invoke(question)