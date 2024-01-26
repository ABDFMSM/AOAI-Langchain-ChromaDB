from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.llms.openai import AzureOpenAI
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
    # We will be passing all files as an argument when running the python script
    paths = list(sys.argv[1:])
    
    for path in paths:
        # Load the file and split it into pages. 
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        # Here we are saving our vector embedding database on the local storage. 
        Chroma.from_documents(pages, embedding_function, persist_directory=r".\chroma_db")

def search_pages(query):
    # Here we are loading our vector embedding database from the local storage. 
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

    # Here we are retrieving the data and chaining it to a completion AI model to summarize.
    rag_chain = RetrievalQA.from_chain_type(
        llm=AzureOpenAI(deployment_name=os.getenv("Completion_model"), openai_api_version="2023-05-15"),
        retriever=db.as_retriever(), 
        return_source_documents=True
    )
    return rag_chain({"query": query})

# Here we check if there are files passed or if the user wants to get informatoin from the already configured chromaDB. 
if len(sys.argv[:]) > 1:
    load_pages()

query = input("What you would like to look for?\n")
result = search_pages(query)

print(result["source_documents"][0].metadata)
print("-------------------------------------------")
print(result["result"])
print("-------------------------------------------")
