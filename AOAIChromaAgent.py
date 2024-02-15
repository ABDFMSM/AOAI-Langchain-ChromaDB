from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI, AzureOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from dotenv import load_dotenv
import os, sys
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

#Load env variable and configure the embedding model
load_dotenv()
embedding_function = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("Embedding_model"),
    openai_api_version="2023-05-15"
    )

def load_pages():
    # We will be passing the file name as an argument when running the python script
    paths = list(sys.argv[1:])
    
    for path in paths:
        # Load the file and split it into pages. 
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        # Here we are saving our vector embedding database on the local storage. 
        Chroma.from_documents(pages, embedding_function, persist_directory=r".\chroma_db")


# Here we are loading our vector embedding database from the local storage. 
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
retriever = db.as_retriever()
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are an AI assistance who can access a vector Database to get answers to customer's questions. 
                       When answering questions try to stick to the information provided by the database as much as you can. 
                       If you don't find an answer to the question, just say "Sorry, I wasn't able to retrieve information from the database, please check the internet".
                       Also, provide a brief one paragraph answer to the question unless the user asks you to list things.  
                       Return the source documents as document name and pages.
                       After answering, ask the user if they have other questions they would like an answer for. 
            """
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        MessagesPlaceholder(
            variable_name='agent_scratchpad'
        ),
        HumanMessagePromptTemplate.from_template(
            "{input}"
        ),  # Where the human input will injected
    ]
)

memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)

llm = AzureChatOpenAI(deployment_name=os.getenv("Chat_model"), openai_api_version="2023-12-01-preview")

chain = RetrievalQA.from_chain_type(
    llm=AzureOpenAI(deployment_name=os.getenv("Completion_model"), 
    openai_api_version="2023-05-15"), 
    chain_type='stuff', 
    retriever=retriever, 
    return_source_documents=True, 
    input_key="question")

tool = Tool(
    name="search_norhtWind_insurance",
    func=lambda query: chain.invoke({"question": query}),
    description="Use it to answer any questions related to insurance."
)

tools = [tool]

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)


def main(): 
    if len(sys.argv[:]) > 1:
        load_pages()

    question = input("What do you like to ask?\n")

    while "exit" not in question: 
        result = agent_executor.invoke({"input": question})
        print(result['output'])
        question = input("\n")

if __name__ == "__main__":
    main()