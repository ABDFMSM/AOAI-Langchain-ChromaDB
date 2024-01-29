from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureOpenAI
from dotenv import load_dotenv
import os 
import sys

#Load env variable and configure the embedding model
load_dotenv()
embedding_function = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("Embedding_model"),
    openai_api_version="2023-05-15"
    )


def load_pages():
    """
    This function loads pages from files provided as command-line arguments and 
    splits them into pages using PyPDFLoader. It then saves the vector embedding 
    database on the local storage using Chroma.from_documents.
    """
    # We will be passing all files as an argument when running the python script
    paths = list(sys.argv[1:])
    for path in paths:
        # Load the file and split it into pages. 
        pages = PyPDFLoader(path).load_and_split()
        # Here we are saving our vector embedding database on the local storage. 
        Chroma.from_documents(pages, embedding_function, persist_directory=r".\chroma_db")

def search_pages(query):
    """
    Search for pages using a specified query and retrieve the data, chaining it to a completion
    AI model to summarize. 

    Args:
        query (str): The query string to search for.

    Returns:
        dict: The summarized data retrieved from the pages.
    """
    # Here we are loading our vector embedding database from the local storage. 
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    # Here we are retrieving the data and chaining it to a completion AI model to summarize.
    rag_chain = RetrievalQA.from_chain_type(
        llm=AzureOpenAI(deployment_name=os.getenv("Completion_model"), openai_api_version="2023-05-15"),
        retriever=db.as_retriever(), 
        return_source_documents=True
    )
    return rag_chain.invoke({"query": query})

def input_query():
    """
    Takes user input, performs a search, and prints the results until the user types "exit".
    """
    query = input("What you would like to look for?\n")
    while "exit" not in query:
        result = search_pages(query)
        print(result["source_documents"][0].metadata)
        print("-------------------------------------------")
        print(result["result"])
        print("-------------------------------------------")
        query = input("What else you would like to look for? if you don't have anything else to look for, type exit\n")

# Here we check if there are files passed or if the user wants to get informatoin from the already configured chromaDB. 
if len(sys.argv[:]) > 1:
    load_pages()

input_query()
