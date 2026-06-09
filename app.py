# import streamlit as st
# from langchain_classic.chains import RetrievalQA
# from src.chatbot import get_chatbot

# st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
# st.title("RAG Chatbot")
# st.info("Using a local HuggingFace model. No OpenAI API key is required.")

# retriever, llm = get_chatbot()
# question = st.text_input("Ask a question")

# if question:
#     with st.spinner("Searching for an answer..."):
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             chain_type="stuff"
#         )
#         answer = qa_chain.run(question)
#         st.write(answer)

# import streamlit as st
# from langchain_classic.chains import RetrievalQA
# from src.chatbot import get_chatbot

# st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# st.title("🤖 RAG Chatbot")
# st.info("Using a local HuggingFace model. No OpenAI API key is required.")

# retriever, llm = get_chatbot()

# question = st.text_input("Ask a question")

# if question:
#     with st.spinner("Searching for an answer..."):

#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             chain_type="stuff",
#             return_source_documents=True
#         )

#         result = qa_chain.invoke({"query": question})

#         st.subheader("Answer")
#         st.write(result["result"])

#         st.subheader("Retrieved Context")

#         for i, doc in enumerate(result["source_documents"], start=1):
#             with st.expander(f"Document Chunk {i}"):
#                 st.write(doc.page_content)


import streamlit as st
from langchain_classic.chains import RetrievalQA
from src.chatbot import get_chatbot
import time

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 RAG Chatbot")
st.info("Using a local HuggingFace model. No OpenAI API key is required.")

@st.cache_resource
def load_qa_chain():
    retriever, llm = get_chatbot()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

qa_chain = load_qa_chain()

question = st.text_input("Ask a question")

if question:
    with st.spinner("Searching for an answer..."):

        start = time.time()
        result = qa_chain.invoke({"query": question})
        end = time.time()

        st.write(f"Response time: {end-start:.2f} sec")

        st.subheader("Answer")
        st.write(result["result"])