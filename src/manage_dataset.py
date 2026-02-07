from langsmith import Client
from src.config import Config

# Initialize LangSmith Client
client = Client()

dataset_name = "devdocs-qa-dataset"

# 10 Comprehensive Question-Answer Pairs (Ground Truth)
examples = [
    {
        "input": "How do I create a StateGraph in LangGraph?",
        "output": "To create a StateGraph, you first define a State class (TypedDict) and then initialize StateGraph(State). Example: `workflow = StateGraph(AgentState)`.",
    },
    {
        "input": "What is the purpose of `.invoke` in LangChain runnables?",
        "output": "The `.invoke` method is used to execute a runnable (chain, model, or tool) synchronously with a single input.",
    },
    {
        "input": "How can I merge two DataFrames in Pandas?",
        "output": "You can use `pd.merge(df1, df2, on='key')` to merge DataFrames based on a common key, similar to SQL joins.",
    },
    {
        "input": "What is Retrieval Augmented Generation (RAG)?",
        "output": "RAG is a technique that combines an LLM with external knowledge retrieval. It retrieves relevant documents from a vector store and passes them as context to the LLM to generate more accurate answers.",
    },
    {
        "input": "How do I add a node to a LangGraph workflow?",
        "output": "Use `workflow.add_node(name, function)` to add a node. The function should take the state as input and return an update to the state.",
    },
    {
        "input": "Explain the difference between `loc` and `iloc` in Pandas.",
        "output": "`loc` is label-based indexing (uses row/column names), while `iloc` is integer-position based indexing (uses 0-based indices).",
    },
    {
        "input": "How do I handle memory in a LangGraph agent?",
        "output": "In LangGraph, you persist state using a Checkpointer (e.g., MemorySaver) and pass it to `.compile(checkpointer=memory)`. This allows resuming threads.",
    },
    {
        "input": "What is FAISS used for?",
        "output": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors, commonly used for vector stores in RAG.",
    },
    {
        "input": "How do I stream tokens from a LangChain model?",
        "output": "You can key use `.stream(input)` on a runnable to get an iterator that yields chunks of the response as they are generated.",
    },
    {
        "input": "What is the `GoogleGenerativeAIEmbeddings` class used for?",
        "output": "It is a LangChain wrapper for Google's embedding models (like embedding-001) to generate vector representations of text.",
    },
    # Acme Corp Handbook Examples
    {
        "input": "What is the maximum expense allowance for a meal at Acme Corp?",
        "output": "The meal allowance is $75 USD per day while traveling for business.",
    },
    {
        "input": "Does Acme Corp allow international remote work?",
        "output": "Yes, but it requires prior approval from HR and is limited to 30 days per calendar year.",
    },
    {
        "input": "What is the Home Office Stipend amount?",
        "output": "All full-time employees are eligible for a one-time Home Office Stipend of $1,500 USD upon joining.",
    },
]

def create_dataset():
    print(f"Checking for existing dataset: {dataset_name}...")
    if client.has_dataset(dataset_name=dataset_name):
        print(f"Dataset {dataset_name} already exists. Deleting to update...")
        client.delete_dataset(dataset_name=dataset_name)
        
    print(f"Creating dataset: {dataset_name}")
    dataset = client.create_dataset(dataset_name=dataset_name, description="QA pairs for DevDocs Navigator RAG Evaluation")
        
    print("Adding examples to dataset...")
    for example in examples:
        client.create_example(
            inputs={"question": example["input"]},
            outputs={"answer": example["output"]},
            dataset_id=dataset.id,
        )
    print("Dataset created successfully!")

if __name__ == "__main__":
    create_dataset()
