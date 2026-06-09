# from langchain_community.llms import HuggingFacePipeline
# from src.vectorstore import load_vectorstore
# from transformers import pipeline

# def get_chatbot():
#     db = load_vectorstore()
#     retriever = db.as_retriever()

#     text_generation = pipeline(
#         "text-generation",
#         model="google/flan-t5-small",
#         do_sample=False,
#         max_new_tokens=256,
#         device=-1
#     )
#     llm = HuggingFacePipeline(pipeline=text_generation)
#     return retriever, llm


from langchain_community.llms import HuggingFacePipeline
from src.vectorstore import load_vectorstore
from transformers import pipeline

def get_chatbot():
    db = load_vectorstore()
    retriever = db.as_retriever()

    text_generation = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=256,
        do_sample=False,
        device=-1
    )

    llm = HuggingFacePipeline(
        pipeline=text_generation
    )

    return retriever, llm