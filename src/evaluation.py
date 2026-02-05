from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config
from src.graph import app as rag_app
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize LangSmith Client
client = Client()

# Initialize Judge LLM
eval_llm = ChatGoogleGenerativeAI(
    model=Config.MODEL_NAME, 
    google_api_key=Config.GOOGLE_API_KEY,
    temperature=0
)

def target(inputs: dict) -> dict:
    """
    Wrapper around RAG app to adapt inputs/outputs for LangSmith evaluation.
    """
    query = inputs["question"]
    # Invoke the RAG pipeline
    result = rag_app.invoke(
        {"question": query, "messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": "eval_thread"}}
    )
    # Extract the final answer from the last message
    answer = result["messages"][-1].content
    # Return dictionary matching the expected output schema for evaluators
    return {"output": answer, "context": result.get("context", "")}

def correctness_evaluator(run, example) -> dict:
    """
    Custom evaluator that checks correctness of the answer vs the ground truth.
    """
    prediction = run.outputs["output"]
    reference = example.outputs["answer"]
    question = example.inputs["question"]

    # Simple LLM-as-a-Judge prompt
    prompt = f"""You are an expert technical documentation grader.
    
    Question: {question}
    Ground Truth Answer: {reference}
    Student Answer: {prediction}

    Grade the Student Answer based on the Ground Truth. 
    It doesn't need to be word-for-word, but must cover the key technical points.
    
    Return a score between 0 and 1, where 1 is correct and 0 is incorrect.
    Also provide a brief reasoning.
    
    Format output as:
    Score: [0-1]
    Reason: [Text]
    """
    
    response = eval_llm.invoke([HumanMessage(content=prompt)]).content
    
    # Simple parsing (robustness could be improved with structured output)
    try:
        import re
        score_match = re.search(r"Score:\s*([0-9.]+)", response)
        score = float(score_match.group(1)) if score_match else 0.0
        return {"key": "correctness", "score": score, "comment": response}
    except:
        return {"key": "correctness", "score": 0.0, "comment": f"Failed to parse grade: {response}"}

def run_evaluation(dataset_name: str = "devdocs-qa-dataset"):
    """
    Run evaluation on a dataset using the target function (the RAG graph).
    """
    print(f"Starting evaluation on dataset: {dataset_name}")
    
    experiment_results = evaluate(
        target,
        data=dataset_name,
        evaluators=[correctness_evaluator],
        experiment_prefix="devdocs-eval",
        metadata={"version": "1.0", "embedding_model": Config.EMBEDDING_MODEL},
        max_concurrency=1
    )
    
    return experiment_results

if __name__ == "__main__":
    results = run_evaluation()
    print("Evaluation Complete. View traces in LangSmith.")
