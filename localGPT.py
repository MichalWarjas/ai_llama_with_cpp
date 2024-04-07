from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import torch
from transformers import (
    GenerationConfig,
    pipeline,
)
from utils import get_embeddings
from templates import get_prompt_template

from constants import (
    PERSIST_DIRECTORY,
    CHROMA_SETTINGS,
)

prompt, history = get_prompt_template("mistral")

# prompt = PromptTemplate.from_template(template)



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
# Chain

embeddings = get_embeddings("cuda" if torch.cuda.is_available() else "cpu")


# load the vectorstore
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever()


qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

while True:
    query = input("\nEnter a query: ")
    if query == "exit":
        break
    # Get the answer from the chain
    res = qa(query)
    answer, docs = res["result"], res["source_documents"]

    # Print the result
    print("\n\n> Question:")
    print(query)
    print("\n> Answer:")
    print(answer)
'''
    if show_sources:  # this is a flag that you can set to disable showing answers.
        # # Print the relevant sources used for the answer
        print("----------------------------------SOURCE DOCUMENTS---------------------------")
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

    # Log the Q&A to CSV only if save_qa is True
    if save_qa:
        utils.log_to_csv(query, answer)

'''
# llm.invoke(question)
