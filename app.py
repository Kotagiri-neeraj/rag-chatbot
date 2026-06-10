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
from src.chatbot import get_chatbot
import time

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 RAG Chatbot")
st.info("Using a local HuggingFace model. No OpenAI API key is required.")

@st.cache_resource
def load_qa_chain():
    retriever, llm = get_chatbot()
    return {
        "retriever": retriever,
        "llm": llm
    }

qa_chain = load_qa_chain()

question = st.text_input("Ask a question")

if question:
    with st.spinner("Searching for an answer..."):

        start = time.time()
        retriever = qa_chain["retriever"]
        llm = qa_chain["llm"]

        # get documents from retriever (use wrapped if adapter)
        try:
            docs = retriever.get_relevant_documents(question)
        except Exception:
            # fallback to wrapped attribute used earlier
            docs = getattr(retriever, "wrapped", retriever).get_relevant_documents(question)

        # build prompt from top documents - ask for comprehensive answer
        context = "\n\n".join([d.page_content for d in docs[:5]])
        prompt = f"""Use the following context to answer the question comprehensively. Provide a detailed and thorough answer with all relevant information.

Context:
{context}

Question: {question}

Comprehensive Answer:""" 

        # call the generate function (returns a plain string)
        output = llm(prompt)
        # generate function returns a string directly
        text = str(output).strip() if output else "(No response generated)"

        end = time.time()

        st.write(f"Response time: {end-start:.2f} sec")

        st.subheader("Answer")
        st.write(text)

        st.subheader("Sources")
        seen = set()
        for doc in docs:
            try:
                meta = getattr(doc, "metadata", {}) or {}
                src = meta.get("source")
                page = meta.get("page")
            except Exception:
                src = None
                page = None

            if not src:
                continue

            key = f"{src}:{page}"
            if key in seen:
                continue
            seen.add(key)

            st.write(f"{src} (Page {page})")