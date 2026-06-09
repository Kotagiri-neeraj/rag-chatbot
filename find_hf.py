import pathlib
root = pathlib.Path(r'c:\Users\HP\Desktop\rag-chatbot\venv\Lib\site-packages')
matches=[]
for p in root.rglob('*.py'):
    try:
        if 'HuggingFacePipeline' in p.read_text(errors='ignore'):
            matches.append(str(p))
    except Exception:
        pass
print(len(matches))
print(matches[:20])
