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

    # Chain-of-Thought Prompt for Correctness
    prompt = f"""You are a strict technical document grader.
    
    Question: {question}
    Ground Truth Answer: {reference}
    Student Answer: {prediction}
    
    Evaluate the Student Answer based on the Ground Truth.
    
    Steps:
    1. Identify key facts in the Ground Truth.
    2. Check if the Student Answer contains these facts.
    3. Check if the Student Answer contains hallucinations or info not in Ground Truth (if strict).
    4. Provide a reasoning for the score.
    
    Score 1.0 if the answer is fully correct and complete.
    Score 0.5 if key facts are missing but the answer is partially correct.
    Score 0.0 if the answer is wrong or irrelevant.
    
    Output strictly in this format:
    Score: [0.0 - 1.0]
    Reason: [Concise explanation]
    """
    
    try:
        response = eval_llm.invoke([HumanMessage(content=prompt)]).content
        
        import re
        score_match = re.search(r"Score:\s*([0-9.]+)", response)
        score = float(score_match.group(1)) if score_match else 0.0
        
        # Extract Reason
        reason_match = re.search(r"Reason:\s*(.+)", response, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else response
        
        return {"key": "correctness", "score": score, "comment": reason}
    except Exception as e:
        return {"key": "correctness", "score": 0.0, "comment": f"Error resolving grade: {e}"}

def faithfulness_evaluator(run, example) -> dict:
    """
    Check if the answer is grounded in the provided context (avoiding hallucinations).
    """
    prediction = run.outputs["output"]
    context = run.outputs["context"]
    
    if not context:
        return {"key": "faithfulness", "score": 0.0, "comment": "No context provided."}
        
    prompt = f"""You are a faithfulness checker.
    
    Context: {context}
    Answer: {prediction}
    
    Is the Answer fully supported by the Context?
    If the answer says "I don't know", score it 1.0 (it is faithful to its lack of knowledge).
    If the answer mentions facts not in context, score 0.0.
    
    Output strictly:
    Score: [0.0 or 1.0]
    Reason: [Explanation]
    """
    
    try:
        response = eval_llm.invoke([HumanMessage(content=prompt)]).content
        
        import re
        score_match = re.search(r"Score:\s*([0-9.]+)", response)
        score = float(score_match.group(1)) if score_match else 0.0
        
        return {"key": "faithfulness", "score": score, "comment": response}
    except:
        return {"key": "faithfulness", "score": 0.0, "comment": "Error parsing faithfulness"}

def run_evaluation(dataset_name: str = "devdocs-qa-dataset"):
    """
    Run evaluation on a dataset using the target function (the RAG graph).
    """
    print(f"Starting evaluation on dataset: {dataset_name}")
    
    experiment_results = evaluate(
        target,
        data=dataset_name,
        evaluators=[correctness_evaluator, faithfulness_evaluator],
        experiment_prefix="devdocs-eval",
        metadata={"version": "1.1", "embedding_model": Config.EMBEDDING_MODEL},
        max_concurrency=1
    )
    
    return experiment_results

if __name__ == "__main__":
    results = run_evaluation()
    print("Evaluation Complete. View traces in LangSmith.")
