from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.ingestion import IngestionEngine
from src.config import Config

# Define State
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    context: str
    question: str

# Initialize Model
# Helper to get LLM
def get_llm():
    if not Config.GOOGLE_API_KEY:
        return None
    return ChatGoogleGenerativeAI(
        model=Config.MODEL_NAME,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0
    )

# Initialize Ingestion for Retrieval
ingestion = IngestionEngine()

def retrieve(state: AgentState):
    """
    Retrieve documents relevant to the question.
    """
    print("---RETRIEVE---")
    question = state["question"]
    
    vectorstore = ingestion.load_vector_store()
    if not vectorstore:
        return {"context": "No documents found. Please run ingestion first."}
        
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(question)
    
    print(f"---DEBUG: Retrieved {len(docs)} docs---")
    for i, d in enumerate(docs):
        print(f"Doc {i} preview: {d.page_content[:100]}...")
        
    context = "\n\n".join([d.page_content for d in docs])
    return {"context": context}

def generate(state: AgentState):
    """
    Generate answer using the retrieved context.
    """
    print("---GENERATE---")
    question = state["question"]
    context = state["context"]
    messages = state["messages"]
    
    # Prompt
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    
    If the answer is not in the context, say "I don't have enough information to answer that based on the provided documentation."
    Always provide citations if possible (though the text might not have explicit URLs, refer to the content).
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = get_llm()
    if not llm:
        return {"messages": [AIMessage(content="Configuration Error: API Key not found.")]}

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    
    return {"messages": [AIMessage(content=response)]}

def route_question(state: AgentState) -> Literal["retrieve", "generate"]:
    """
    Route question to retrieval or generation.
    For this simple version, we always retrieve if it looks like a technical question.
    """
    # Simple logic: always retrieve for now as it's a documentation assistant
    return "retrieve"

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Entry point
workflow.set_entry_point("retrieve")

# Edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()
