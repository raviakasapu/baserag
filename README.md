---
title: Base Rag q and A
emoji: 🌍
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: 1.28.2
app_file: app.py
pinned: false
license: apache-2.0
---

# LLM Evluation using Ragas and Langchain

Ragas is a framework that helps you evaluate an enterprise Retrieval Augmented Generation (RAG) pipelines. 
Ragas is very easy to use and evaluate the RAG since there is no additional data required. The Context used in the RAG pipeline and  Question and Answers are used for evaluating the RAG.

Ragas can provide below metrics https://docs.ragas.io/en/latest/concepts/metrics/index.html

* Faithfulness
* Answer relevancy
* Context recall
* Context precision
* Context relevancy
* Context entity recall

We will use LangChain framework to implement the RAG and functions/chains provided within LangChain

## Purpose

Evaluation or RAG approach using LangChain and OpenAI

## Features


## Usage
 * add your PDF files in the data folder
 * update the path in the vector_loader.py and run the file using
    `python vector_loader.py`
* update the index name for the DB
* this will generate local FAISS vector db files
* update the index files in app.py 
* run the streamlit app using
    `streamlit run app.py`

## Sample Output


## Future Enhancements

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug fixes, please submit a pull request or open an issue in the GitHub repository.

## License

This project is licensed under the MIT License.
