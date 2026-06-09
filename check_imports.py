import sys
try:
    from langchain_classic.chains import RetrievalQA
    print('RetrievalQA ok', RetrievalQA)
except Exception as e:
    print('RetrievalQA fail', type(e).__name__, e)
try:
    from langchain_classic.llms import HuggingFacePipeline
    print('HuggingFacePipeline ok', HuggingFacePipeline)
except Exception as e:
    print('HuggingFacePipeline fail', type(e).__name__, e)
