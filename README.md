# 🧭 DevDocs Navigator

**DevDocs Navigator** is an AI-powered documentation assistant designed to help developers navigate complex technical documentation. It uses **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware answers based on the official documentation of libraries like **LangChain**, **LlamaIndex**, and **Pandas**.

## 🚀 Key Features

*   **RAG Pipeline**: Retrieves relevant context from ingested documentation to answer user queries accurately.
*   **Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` (via HuggingFace) for generating embeddings locally, ensuring **no rate limits** and **zero cost** for embedding generation.
*   **Vector Search**: Utilizes **FAISS** for efficient similarity search and retrieval.
*   **Persistent Knowledge Base**: The vector store is saved locally (`faiss_index/`), so you only need to ingest data once.
*   **Streamlit UI**: A clean, interactive user interface for managing data ingestion and chatting with the AI.
*   **Google Gemini Powered**: Uses Google's **Gemini 1.5 Flash** (or similar) models for high-quality response generation.

## 🛠️ Tech Stack

*   **Python 3.10+**
*   **LangChain & LangGraph**: Orchestration and stateful agent workflows.
*   **Streamlit**: Frontend UI.
*   **FAISS**: Vector Database.
*   **HuggingFace**: Local Embeddings.
*   **Google Generative AI**: LLM Provider.

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/prnav1923/dev-doc.git
cd dev-doc
```

### 2. Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory (copy from `.env.example`):
```bash
cp .env.example .env
```
Edit `.env` and add your API keys:
```env
GOOGLE_API_KEY=your_google_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=devdocs-navigator
# LANGSMITH_API_KEY=your_langsmith_key (Optional, for tracing)
```

## 🏃‍♂️ Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

## 💡 Usage

1.  **Ingest Data (First Run Only)**:
    *   On the first launch, the sidebar will show "⚠️ Knowledge Base: Not Found".
    *   Click the **"Re-ingest / Update Documentation"** button.
    *   Wait for the process to complete (this downloads docs and builds the local vector index).
    *   Once done, you'll see "✅ Knowledge Base: Ready".

2.  **Ask Questions**:
    *   Type your question in the chat input (e.g., *"How do I create a graph in LangGraph?"*).
    *   The agent will retrieve relevant docs and generate an answer with citations.

3.  **Persistence**:
    *   The next time you run the app, the Knowledge Base will be ready instantly. You only need to run ingestion again if you want to update the documentation.

## 🧪 Evaluation (LLM-as-a-Judge)

The project includes a built-in evaluation pipeline using **LangSmith** and **Gemini** to test the RAG system's accuracy.

1.  **Seed Ground Truth Data**:
    ```bash
    python src/manage_dataset.py
    ```
    This creates a dataset `devdocs-qa-dataset` in LangSmith with 10 sample Q&A pairs.

2.  **Run Evaluation**:
    *   In the Streamlit App Sidebar, click **"Run LangSmith Evaluation"**.
    *   The system will compare the RAG's answers against the ground truth using an LLM to grade correctness.
    *   View results in the [LangSmith Datasets](https://smith.langchain.com/datasets) dashboard.

## 🤝 Contributing

Contributions are welcome! Please feel free to verify the `feature/initial-setup` branch or open a Pull Request.
