import os
import sys

import openai
import config

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from transformers import pipeline
import gradio as gr


def transcribe_audio(mic):

    transcription = audio_model(mic)["text"]
    return transcription


def ask_question(question = None, mic = None):
    if mic is None:
        input_text = question
    else:
        input_text = transcribe_audio(mic=mic)
        
    result = qa({"query": input_text})
    
    res = """
    Question: {}\n
    Answer: {}\n
    """.format(input_text, result['result'])
    
    return res


if __name__ == "__main__":
    openai.api_key  = config.OPENAI_KEY
    os.environ['OPENAI_API_KEY'] = config.OPENAI_KEY


    audio_model = pipeline("automatic-speech-recognition")

    LLM_NAME = "gpt-3.5-turbo"

    llm = ChatOpenAI(model_name=LLM_NAME, temperature=0)
    persist_directory = './chroma'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    template = """
    The following pieces of texts contain fragments from RITIS tutorials.
    Use these texts to addres the question and give the correct answer. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use ten sentences maximum. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    question = "What tool can I use to rank bottlenecks?"
    qa= RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    with open('app_description.html', 'r') as f:
        app_description = f.read()

    gr.Interface(
        fn=ask_question,
        inputs=[
            gr.inputs.Textbox(label="Question", optional=True),
            gr.Audio(source="microphone", type="filepath", optional=True),
        ],
        outputs="text",
        
        title="RITIS chatbot - v 0.01",  
        description=app_description,
        
        
        examples = [
            ["What tool can I use to rank bottlenecks?", None],
            ["how can I ask for CO2 emissions in College Park", None],
            ["How to estimate the delay costs", None],
            ["I need to analyze carbon dioxide emissions " \
                "in 2020 at I-95 corridor and check if there " \
                "were any anomalies. Can I do it with RITIS?", None],
            ["Give me all the tools that can analyze carbon dioxide emissions.", None],
        ]
    ).launch(server_name="0.0.0.0", server_port=8891)
