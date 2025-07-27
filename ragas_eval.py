import json
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerCorrectness, ContextPrecision
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("api_key")
# === Load dataset ===
with open("rag_eval_dataset_medquad.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# === Setup LLM and embeddings ===
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key=OPENAI_API_KEY)
evaluator_llm = LangchainLLMWrapper(llm)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Prepare filtered evaluation dataset ===
transformed_filtered = [
    {
        "user_input": item["question"],
        "retrieved_contexts": item["retrieved_chunks_filtered"],
        "response": item["generated_answer_filtered"],
        "reference": item["expected_answer"]
    }
    for item in raw_data
]

dataset_filtered = EvaluationDataset.from_list(transformed_filtered)

# === Prepare unfiltered evaluation dataset ===
transformed_unfiltered = [
    {
        "user_input": item["question"],
        "retrieved_contexts": item["retrieved_chunks_unfiltered"],
        "response": item["generated_answer_unfiltered"],
        "reference": item["expected_answer"]
    }
    for item in raw_data
]

dataset_unfiltered = EvaluationDataset.from_list(transformed_unfiltered)

# === Evaluate both ===
print("🔍 Evaluating filtered retrieval...")
result_filtered = evaluate(
    dataset=dataset_filtered,
    metrics=[ContextPrecision(), AnswerCorrectness()],
    llm=evaluator_llm,
    embeddings=embeddings
)
print("✅ Filtered Results:", result_filtered)

print("\n🔍 Evaluating unfiltered retrieval...")
result_unfiltered = evaluate(
    dataset=dataset_unfiltered,
    metrics=[ContextPrecision(), AnswerCorrectness()],
    llm=evaluator_llm,
    embeddings=embeddings
)
print("✅ Unfiltered Results:", result_unfiltered)