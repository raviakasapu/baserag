from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

import PyPDF2
from PyPDF2 import PdfReader
import pdfplumber
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as pdf_file:
        # Read the PDF file
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        # Get the number of pages in the PDF
        num_pages = len(pdf_reader.pages)
        # Initialize an empty string to store the text
        full_text = ''
        # Loop through each page and extract the text
        for page_num in range(num_pages):
            # Get the page object
            #page = PyPDF2.PdfReader()
            # Extract the text from the page
            page_text = pdf_reader.pages[page_num].extract_text()
            # Append the text to the full_text variable
            full_text += page_text
    # Return the full text of the PDF
    return full_text

model = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
embeddings = HuggingFaceEmbeddings(model_name = model)

def save_to_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, 
                                               chunk_overlap=20,
                                               length_function=len,
                                               is_separator_regex=False)
    docs = text_splitter.create_documents([text])
    vectorstore = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY))
    #vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH, index_name="njmvc_Index")
#create a new file named vectorstore in your current directory.
if __name__=="__main__":
        DB_FAISS_PATH = './vectorstore/db_faiss/'
        file_name = "./data/drivermanual-2-small.pdf"
        #loader=read_file_get_prompts(file_name)
        #text=read_file_get_prompts(file_name)
        text = extract_text_from_pdf(file_name)
        #pdfReaded = PyPDF2.PdfReader(file_name)
        #docs=loader.load()
        #save_to_vector_store(text)
        save_to_vector_store(text)
        
        