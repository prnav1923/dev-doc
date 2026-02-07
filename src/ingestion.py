import os
from typing import List
from langchain_community.document_loaders import WebBaseLoader, RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import Config

class IngestionEngine:
    def __init__(self):
        # Use local embeddings to avoid API rate limits
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_urls(self, urls: List[str]):
        """Load content from a list of specific URLs."""
        docs = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {url}: {e}")
        return docs

    def load_pdf(self, file_path):
        """Load a PDF file."""
        from langchain_community.document_loaders import PyPDFLoader
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return []
            
    def load_text(self, file_path):
        """Load a text/markdown file."""
        from langchain_community.document_loaders import TextLoader
        try:
            loader = TextLoader(file_path)
            return loader.load()
        except Exception as e:
            print(f"Error loading text {file_path}: {e}")
            return []

    def load_directory(self, directory_path):
        """Load all supported files from a directory."""
        docs = []
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return docs
            
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith(".pdf"):
                    docs.extend(self.load_pdf(file_path))
                elif file.lower().endswith((".txt", ".md")):
                    docs.extend(self.load_text(file_path))
        return docs

    def process_documents(self, docs):
        """Split documents into chunks."""
        splits = self.text_splitter.split_documents(docs)
        return splits

    def create_vector_store(self, splits):
        """Create and save FAISS vector store with rate limit handling."""
        if not splits:
            print("No documents to index.")
            return None
            
        vectorstore = None
        batch_size = 5 # Increased slightly as local embeddings are fast, but FAISS might need care? 
                       # Actually the rate limit was for embedding API if used, but we use local HuggingFace.
                       # Wait, we use HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") which is local.
                       # The previous code had rate limit logic likely because the user MIGHT switch to OpenAI/Gemini embeddings.
                       # I'll keep the logic but it's less critical for local embeddings.
                       
        import time

        print(f"Ingesting {len(splits)} chunks in batches of {batch_size}...")
        
        # Simple loop without tqdm
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i+batch_size]
            retries = 3
            success = False
            for attempt in range(retries):
                try:
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(documents=batch, embedding=self.embeddings)
                    else:
                        new_vectorstore = FAISS.from_documents(documents=batch, embedding=self.embeddings)
                        vectorstore.merge_from(new_vectorstore)
                    
                    # Small buffer
                    # time.sleep(0.5) 
                    success = True
                    break # Success, move to next batch
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait_time = (attempt + 1) * 5
                        print(f"Rate limit hit on batch {i}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error processing batch {i}: {e}")
                        break # Non-retryable error
            
            if not success:
                print(f"Failed to process batch {i} after {retries} retries. Aborting.")
                return None

        if vectorstore:
            vectorstore.save_local(Config.VECTOR_STORE_PATH)
            print(f"Vector store saved to {Config.VECTOR_STORE_PATH}")
        else:
            print("Failed to create vector store.")
            
        return vectorstore

    def load_vector_store(self):
        """Load existing vector store."""
        if os.path.exists(Config.VECTOR_STORE_PATH):
            return FAISS.load_local(
                Config.VECTOR_STORE_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return None

if __name__ == "__main__":
    # Test ingestion
    ingestion = IngestionEngine()
    
    # 1. Load URLs
    print("Loading Web Docs...")
    sample_urls = [
         "https://python.langchain.com/docs/get_started/introduction",
    ]
    web_docs = ingestion.load_urls(sample_urls)
    
    # 2. Load Local Data (if exists)
    print("Loading Local Data...")
    local_docs = ingestion.load_directory("data")
    
    all_docs = web_docs + local_docs
    print(f"Total Loaded {len(all_docs)} documents.")
    
    if all_docs:
        print("Splitting docs...")
        splits = ingestion.process_documents(all_docs)
        print(f"Created {len(splits)} chunks.")
        
        print("Creating vector store...")
        ingestion.create_vector_store(splits)
