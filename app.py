import streamlit as st  #Web App
import numpy as np #Image Processing 
import pandas as pd

import time

import os
import tiktoken
from io import StringIO
import time
import json

import requests
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv,find_dotenv

#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from dotenv import load_dotenv
from htmlTemplates import bot_template, user_template, css

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def load_knowledgeBase():
    embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    DB_FAISS_PATH = "./vectorstore/db_faiss/"
    db = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True,
            index_name="njmvc_Index"
        )   
    return db
def load_prompt():
    prompt = """ You are helping students to pass NJMVC Knowledge Test. Provide a Single multiple choice question with 4 options to choose from.
    Use the context to provide the question and answer choices.
    context = {context}
    question = {question}
    if the answer is not in the pdf answer "i donot know what the hell you are asking about"
        """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt

#function to load the OPENAI LLM
def load_llm():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
    return llm

knowledgeBase=load_knowledgeBase()
prompt = load_prompt()
llm=load_llm()

def get_conversation_chain(vectorstore, llm):
        llm = llm
        #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

        memory = ConversationBufferMemory(memory_key="chat_history")
        conversation_chain = ConversationChain(
                llm=llm,
                verbose=True,
                memory=ConversationBufferMemory(),
        )
        return conversation_chain

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def get_pdf_text(pdf_files):
    
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def handle_user_input(question):
        response = st.session_state.conversation({'question':question})

        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                        st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                        st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title='NJMVC Knowledge Test with RAGAS', page_icon=':cars:')

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('NJMVC Knowledge Test with RAGAS :cars:')
    question = st.text_input("Input the Topic you want to test your knowledge: ")

    if question:
        #handle_user_input(question)

        with st.spinner("Get ready..."):  
            text_chunks = get_chunk_text(question)
            
            db = FAISS.load_local(folder_path="./vectorstore/db_faiss/",embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY),allow_dangerous_deserialization=True, index_name="njmvc_Index")
            searchDocs = db.similarity_search("what is the NJMVC driving test")

            similar_embeddings=FAISS.from_documents(documents=searchDocs, embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY))
            #creating the chain for integrating llm,prompt,stroutputparser
            retriever = similar_embeddings.as_retriever()
            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
            )
            #st.session_state.conversation = get_conversation_chain(vector_store)
            
            response=rag_chain.invoke(question)
            st.write(response)

if __name__ == '__main__':
    main()
