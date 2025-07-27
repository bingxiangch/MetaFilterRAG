import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import SentenceTransformer

# === Load environment variables ===
load_dotenv()
collection_name = os.getenv("collection_name")
qdrant_url = os.getenv("qdrant_url")
qdrant_api_key = os.getenv("qdrant_api_key")
openai_api_key = os.getenv("api_key")

# === Setup clients ===
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
openai_client = OpenAI(api_key=openai_api_key)

# === Load QA data ===
with open("xxx.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

results = []

# === System prompt for answer generation ===
system_prompt = (
    "You are a helpful medical assistant. Use ONLY the provided context to answer the question.\n"
    "If the answer is not found in the context, respond exactly with:\n"
    "\"The answer is not available in the provided context.\""
)

for idx, qa in enumerate(qa_pairs):
    question = qa["question"]
    expected_answer = qa["answer"]
    qtypes = qa.get("qtype", [])
    qtypes = [qtypes] if isinstance(qtypes, str) else qtypes

    # === Step 1: Embed the question ===
    vector = embedding_model.encode(question).tolist()

    # === Step 2: Build Qdrant filter ===
    query_filter = Filter(
        should=[FieldCondition(key="qtype", match=MatchText(text=q)) for q in qtypes]
    ) if qtypes else None

    # === Step 3: Retrieve with filter ===
    try:
        filtered = qdrant_client.query_points(
            collection_name=collection_name,
            query=vector,
            query_filter=query_filter,
            limit=5
        )
        filtered_chunks = [
            f"Q: {hit.payload.get('question', '').strip()}\nA: {hit.payload.get('answer', '').strip()}"
            for hit in filtered.points if hit.payload.get("answer")
        ]
        context_filtered = "\n\n".join(filtered_chunks)
    except Exception as e:
        print(f"[Error] Filtered retrieval failed for index {idx}: {e}")
        filtered_chunks = []
        context_filtered = ""

    # === Step 4: Retrieve without filter ===
    try:
        unfiltered = qdrant_client.query_points(
            collection_name=collection_name,
            query=vector,
            query_filter=None,
            limit=5
        )
        unfiltered_chunks = [
            f"Q: {hit.payload.get('question', '').strip()}\nA: {hit.payload.get('answer', '').strip()}"
            for hit in unfiltered.points if hit.payload.get("answer")
        ]
        context_unfiltered = "\n\n".join(unfiltered_chunks)
    except Exception as e:
        print(f"[Error] Unfiltered retrieval failed for index {idx}: {e}")
        unfiltered_chunks = []
        context_unfiltered = ""

    # === Step 5: Generate answer (filtered) ===
    try:
        completion_filtered = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_filtered}\n\nQuery:\n{question}"}
            ]
        )
        answer_filtered = completion_filtered.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Error] OpenAI generation (filtered) failed for index {idx}: {e}")
        answer_filtered = "ERROR"

    # === Step 6: Generate answer (unfiltered) ===
    try:
        completion_unfiltered = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_unfiltered}\n\nQuery:\n{question}"}
            ]
        )
        answer_unfiltered = completion_unfiltered.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Error] OpenAI generation (unfiltered) failed for index {idx}: {e}")
        answer_unfiltered = "ERROR"

    # === Step 7: Save result ===
    results.append({
        "question": question,
        "expected_answer": expected_answer,
        "qtype": qtypes,
        "retrieved_chunks_filtered": filtered_chunks,
        "generated_answer_filtered": answer_filtered,
        "retrieved_chunks_unfiltered": unfiltered_chunks,
        "generated_answer_unfiltered": answer_unfiltered
    })

    print(f"Processed {idx + 1}/{len(qa_pairs)}")

# === Final Save ===
with open("rag_eval_dataset_medquad.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Done! Results saved to rag_eval_dataset_medquad.json")
