# RITIS Chat Bot
---

This is the initial work on RITIS Chat Bot. This bot is suppose to be an assistant that guides people through their daily work.

### Installation
Create your own conda environment, for example:
```bash
conda create --name ritis python==3.11
conda activate ritis
```

Install the requirements with
```bash
pip install -r requirements.txt
```

Create `config.py` in parent directory. If you want to use ChatGPT you should define proper `OPENAI_KEY` in this file. If you want to debug your pipelines with LangChain, you should define proper `LANGCHAIN_KEY` (normally not required)

Example of `config.py`:
```bash
OPENAI_KEY = '<YOUR KEY HERE>'
LANGCHAIN_KEY = '<YOUR KEY HERE>'
```

Run the `01_CreateVectorStoreOpenAI.ipynb` notebook to build a database.

Run the chatbot with:
```bash
python ritis_chatbot.py
```

### Content
- `Main folder`:
    - Code for creating vector stores (for now: `01_CreateVectorStoreOpenAI.ipynb`)
    - Code that runs a chatbot (for now: `ritis_chatbot.py`)
    - `config.py` (not stored on GitHub)
- `chroma/`: folder with a vector store. Not stored on GitHub. Generated automatically
- `flagged/`: Folder with gradio flags. Not stored on GitHub. Generated automatically
- `input_texts/`: Input data 
- `tests`: Various notebooks for test purposes


### TODO:
- Add memory
- Create ChatGPT-independent versions



