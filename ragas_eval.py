from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, ResponseRelevancy, ContextPrecision
import json
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
# ⚠️ Don't hardcode secrets in code! Use environment variables instead in real projects.
OPENAI_API_KEY=os.getenv("api_key")
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# Load dataset from JSON file
with open("query_outputs/query_results.json", "r") as f:
    raw_data = json.load(f)

# Transform the dataset to match RAGAS schema
transformed_data = []
for item in raw_data:
    transformed_data.append({
        "user_input": item["query"],
        "retrieved_contexts": item["retrieved_context"],
        "response": item["answer"],
        "reference": item["expected_answer"]
    })
# Create RAGAS evaluation dataset
evaluation_dataset = EvaluationDataset.from_list(transformed_data)

# Prepare evaluator LLM
evaluator_llm = LangchainLLMWrapper(llm)

# Run evaluation
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[ContextPrecision(),ResponseRelevancy()],
    llm=evaluator_llm,
    embeddings=embeddings
)

print("result_with_filter:", result)

# Transform the dataset to match RAGAS schema
transformed_data_withou_filter = []
for item in raw_data:
    transformed_data_withou_filter.append({
        "user_input": item["query"],
        "retrieved_contexts": item["retrieved_context_without_filter"],
        "response": item["answer_without_filter"],
        "reference": item["expected_answer"]
    })
# Create RAGAS evaluation dataset
evaluation_dataset_without_filter = EvaluationDataset.from_list(transformed_data_withou_filter)
# Run evaluation
result_without_filter = evaluate(
    dataset=evaluation_dataset_without_filter,
    metrics=[ContextPrecision(),ResponseRelevancy()],
    llm=evaluator_llm,
    embeddings=embeddings
)

print("result_without_filter:", result_without_filter)

