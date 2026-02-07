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
                   "https://langchain-ai.github.io/langgraph/concepts/high_level/", # LangGraph Concepts
                ]
                docs = ingestion.load_urls(sample_urls)
                splits = ingestion.process_documents(docs)
                ingestion.create_vector_store(splits)
                st.success(f"Successfully ingested {len(splits)} chunks!")
                st.rerun() # Refresh status
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

    st.markdown("---")
    st.subheader("Add Custom Data")
    
    # 1. Upload File
    uploaded_files = st.file_uploader("Upload PDF or Text files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploaded_files and st.button("Process Uploaded Files"):
        with st.spinner("Processing files..."):
            try:
                import tempfile
                ingestion = IngestionEngine()
                all_docs = []
                
                # Create a temp directory to save uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        if uploaded_file.name.endswith(".pdf"):
                            all_docs.extend(ingestion.load_pdf(temp_path))
                        else:
                            all_docs.extend(ingestion.load_text(temp_path))
                    
                    if all_docs:
                        splits = ingestion.process_documents(all_docs)
                        ingestion.create_vector_store(splits)
                        st.success(f"Successfully ingested {len(all_docs)} documents ({len(splits)} chunks)!")
                        st.rerun()
                    else:
                        st.warning("No valid content found in uploaded files.")
            except Exception as e:
                st.error(f"Error processing upload: {e}")

    # 2. Demo Data
    if st.button("Load Demo Data (Acme Corp Handbook)"):
        with st.spinner("Loading Demo Data..."):
            try:
                ingestion = IngestionEngine()
                data_path = os.path.join(Config.PROJECT_ROOT, "data")
                docs = ingestion.load_directory(data_path)
                if docs:
                    splits = ingestion.process_documents(docs)
                    ingestion.create_vector_store(splits)
                    st.success(f"Successfully loaded Acme Corp Handbook ({len(splits)} chunks)!")
                    st.rerun()
                else:
                    st.error("Demo data not found in 'data/' directory.")
            except Exception as e:
                st.error(f"Error loading demo data: {e}")

    st.header("Evaluation")
    if st.button("Run LangSmith Evaluation"):
        with st.spinner("Running evaluation against ground truth dataset..."):
            try:
                from src.evaluation import run_evaluation
                results = run_evaluation()
                st.success("Evaluation Complete!")
                st.markdown(f"View details in [LangSmith Datasets](https://smith.langchain.com/datasets)")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

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
                # Use the session-specific thread_id
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
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
