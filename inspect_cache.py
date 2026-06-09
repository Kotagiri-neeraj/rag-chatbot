import os
from pathlib import Path
cache=os.getenv('TRANSFORMERS_CACHE') or os.getenv('HF_HOME')
print('TRANSFORMERS_CACHE', cache)
root=Path(cache) if cache else Path.home()/'.cache'/'huggingface'/'transformers'
print('cache root', root)
for p in root.rglob('*flan-t5-small*'):
    print(p)
