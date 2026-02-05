import streamlit as st
import os
from langchain_core.messages import HumanMessage
from src.ingestion import IngestionEngine
from src.graph import app as rag_app
from src.config import Config

st.set_page_config(page_title="DevDocs Navigator", layout="wide")

st.title("🧭 DevDocs Navigator")
st.markdown("AI-Powered Documentation Assistant for LangChain, LlamaIndex, and Pandas")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = Config.GOOGLE_API_KEY
    if not api_key:
        st.error("GOOGLE_API_KEY not found in environment variables!")
    elif api_key.startswith("your_"):
        st.error("It looks like you are using the placeholder API key. Please update your .env file with a real key.")
    else:
        st.success(f"API Key loaded (starts with: {api_key[:4]}...)")

    st.header("Data Management")
    
    # Check if vector store exists
    if os.path.exists(Config.VECTOR_STORE_PATH):
        st.success("✅ Knowledge Base: Ready")
    else:
        st.warning("⚠️ Knowledge Base: Not Found. Please ingest data.")
        
    if st.button("Re-ingest / Update Documentation"):
        with st.spinner("Ingesting documentation... This may take a while."):
            try:
                ingestion = IngestionEngine()
                # Use sample URLs for now, in real app might allow user input or broader crawl
                sample_urls = [
                   "https://python.langchain.com/docs/introduction/", 
                   "https://docs.llamaindex.ai/en/stable/",
                   "https://pandas.pydata.org/docs/user_guide/10min.html" 
                ]
                docs = ingestion.load_urls(sample_urls)
                splits = ingestion.process_documents(docs)
                ingestion.create_vector_store(splits)
                st.success(f"Successfully ingested {len(splits)} chunks!")
                st.rerun() # Refresh status
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How do I use LangGraph?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the LangGraph app
                config = {"configurable": {"thread_id": "1"}}
                result = rag_app.invoke(
                    {"question": prompt, "messages": [HumanMessage(content=prompt)]}, 
                    config=config
                )
                
                # Extract the last AI message
                response = result["messages"][-1].content
                st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Debugging: Show retrieved context
                with st.expander("Debug: Retrieved Context"):
                    st.write(result.get("context", "No context returned"))
                
                # Check for context/sources in state if available (not explicitly returned in simple graph but in state)
                # In a real app we would extract documents from state if persisted or returned
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.markdown("Please make sure you have set up the API keys and ingested data.")
