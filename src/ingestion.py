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
        batch_size = 2
        import time

        print(f"Ingesting {len(splits)} chunks in batches of {batch_size}...")
        
        # Simple loop without tqdm
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i+batch_size]
            retries = 5
            success = False
            for attempt in range(retries):
                try:
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(documents=batch, embedding=self.embeddings)
                    else:
                        new_vectorstore = FAISS.from_documents(documents=batch, embedding=self.embeddings)
                        vectorstore.merge_from(new_vectorstore)
                    
                    # Rate limit buffer - start small
                    time.sleep(5) 
                    success = True
                    break # Success, move to next batch
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait_time = (attempt + 1) * 30  # Slower backoff: 30s, 60s, 90s...
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
    # Test ingestion with some sample URLs
    ingestion = IngestionEngine()
    
    # We will use some high-value representative pages for the demo
    # In a real scenario, we might use RecursiveUrlLoader for full sections
    sample_urls = [
        "https://python.langchain.com/docs/get_started/introduction",
        "https://docs.llamaindex.ai/en/stable/",
        "https://pandas.pydata.org/docs/user_guide/index.html"
    ]
    
    print("Loading docs...")
    docs = ingestion.load_urls(sample_urls)
    print(f"Loaded {len(docs)} documents.")
    
    print("Splitting docs...")
    splits = ingestion.process_documents(docs)
    print(f"Created {len(splits)} chunks.")
    
    print("Creating vector store...")
    ingestion.create_vector_store(splits)
