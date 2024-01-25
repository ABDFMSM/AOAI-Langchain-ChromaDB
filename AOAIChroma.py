from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.llms.openai import AzureOpenAI
from dotenv import load_dotenv
import os 
import sys


# We will be passing the file name as an argument when running the python script
path = sys.argv[1]

# Load the file and split it into pages. 
loader = PyPDFLoader(path)
pages = loader.load_and_split()

#Load env variable and configure the embedding model
load_dotenv()
embedding_function = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("Embedding_model"),
    openai_api_version="2023-05-15"
    )

# Here we are saving our vector embedding database on the local storage. 
db = Chroma.from_documents(pages, embedding_function, persist_directory=r".\chroma_db")

query = "Tell me about Women's Health and Cancer Rights Act of 1998?"

# Here we are loading our vector embedding database from the local storage. 
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)


rag_chain = RetrievalQA.from_chain_type(
    llm=AzureOpenAI(deployment_name=os.getenv("Completion_model"), openai_api_version="2023-05-15"),
    retriever=db3.as_retriever(), 
    return_source_documents=True
)

result = rag_chain({"query": query})
print(result["source_documents"][0].metadata['page'])
print("-------------------------------------------")
print(result["result"])
print("-------------------------------------------")
