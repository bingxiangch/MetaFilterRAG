from uuid import uuid4
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, Integer, String
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from openai import OpenAI
import os
import json

# Load environment variables
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
collectionName = os.getenv("collection_name")
qdrant_url = os.getenv("qdrant_url")
# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./sqlite.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, autoflush=False)

# OpenAI + Qdrant setup
openai_client = OpenAI(api_key=os.getenv("api_key"))
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=os.getenv("qdrant_api_key")
)

system_prompt = """
Extract the following metadata from the user's query and return it in valid JSON format. 

- `patient_id`: A list of full patient name(s), e.g., ["Barbara Cantu"].
- `department`: a list of department,  e.g., [Cardiology"].
- `section`: A list of section,  e.g., ["Basic Information"]:
  - "Basic Information"
  - "Current Condition"
  - "Treatment"
  - "Medical History"
  - "Lab Results"
  - "Billing Information"

Only include fields that are explicitly mentioned or clearly implied in the query. Omit any field that is missing or ambiguous.

Query: "{query}"
"""

answer_system_prompt = (
    "You are a helpful medical assistant. Answer only using the context. "
    "If no answer, say 'No information found in the context.'"
)
def query_pipeline(query: str, output_dir: str = "query_outputs"):
    # Step 1: Extract metadata from the query
    meta_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    metadata = json.loads(meta_response.choices[0].message.content.strip())

    # Step 2: Build filter conditions based on metadata
    filter_conditions = []

    # Handle patient_id as "should"
    if "patient_id" in metadata:
        should_conditions = [
            models.FieldCondition(
                key="patient_id",
                match=models.MatchText(text=patient_id)
            ) for patient_id in metadata["patient_id"]
        ]
        if should_conditions:
            filter_conditions.append(
                models.Filter(should=should_conditions)
            )

    # Handle section as individual conditions
    if "section" in metadata:
        for section_value in metadata["section"]:
            filter_conditions.append(
                models.FieldCondition(
                    key="section",
                    match=models.MatchText(text=section_value)
                )
            )

    # Handle department if available
    if "department" in metadata:
        for dept in metadata["department"]:
            filter_conditions.append(
                models.FieldCondition(
                    key="department",
                    match=models.MatchText(text=dept)
                )
            )

    # Combine all filters
    query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

    # Step 3: Retrieve context from Qdrant
    search_result = qdrant_client.query(
        collection_name=collectionName,
        query_text=query,
        limit=10,
        query_filter=query_filter
    )
    retrieve_context = [result.document for result in search_result]
    print('retrieve_context: ', retrieve_context)
    retrieve_context_without = qdrant_client.query(
        collection_name=collectionName,
        query_text=query,
        limit=10,
        query_filter=None
    )
    retrieve_context_without_filter = [result1.document for result1 in retrieve_context_without]
    print('retrieve_context_without: ', retrieve_context_without_filter)

    context = "\n\n".join([hit.document for hit in search_result])
    context_without = "\n\n".join([hit.document for hit in retrieve_context_without_filter])

    # Step 4: Generate final answer using context

    final_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": answer_system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nPlease refer to this context to answer the following query."},
            {"role": "user", "content": f"Query:\n{query}\n\nPlease provide a detailed, accurate response based on the context."}
        ]
    )

    final_answer = final_response.choices[0].message.content.strip()

    # Step 5: Save everything to a JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_data = {
        "query": query,
        "retrieved_context": retrieve_context,
        "generated_answer": final_answer
    }

    output_path = os.path.join(output_dir, f"query_result_{uuid4().hex[:8]}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    return output_data

# result = query_pipeline("What is Layla Harris's current treatment plan?")
# print(result["generated_answer"])

count = 1
def run_batch_query_pipeline(input_file: str, output_dir: str = "query_outputs"):
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
    except Exception as e:
        print(f"Failed to read input file: {e}")
        return []

    results = []
    count = 1
    for item in qa_data:
        for qa in item.get("qa_pairs", []):
            query = qa.get("question")
            expected_answer = qa.get("answer")
            try:
                # Step 1: Extract metadata
                meta_response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ]
                )
                metadata = json.loads(meta_response.choices[0].message.content.strip())
            except Exception as e:
                print(f"Metadata extraction failed for query '{query}': {e}")
                continue

            # Step 2: Build filter
            filter_conditions = []
            try:
                if "patient_id" in metadata:
                    should_conditions = [
                        models.FieldCondition(
                            key="patient_id",
                            match=models.MatchText(text=pid)
                        ) for pid in metadata["patient_id"]
                    ]
                    if should_conditions:
                        filter_conditions.append(models.Filter(should=should_conditions))

                if "section" in metadata:
                    for section_value in metadata["section"]:
                        filter_conditions.append(
                            models.FieldCondition(
                                key="section",
                                match=models.MatchText(text=section_value)
                            )
                        )

                if "department" in metadata:
                    for dept in metadata["department"]:
                        filter_conditions.append(
                            models.FieldCondition(
                                key="department",
                                match=models.MatchText(text=dept)
                            )
                        )

                query_filter = models.Filter(must=filter_conditions) if filter_conditions else None
            except Exception as e:
                print(f"Filter construction failed for query '{query}': {e}")
                continue

            # Step 3: Retrieve context
            try:
                search_result = qdrant_client.query(
                    collection_name=collectionName,
                    query_text=query,
                    limit=10,
                    query_filter=query_filter
                )
                retrieve_context = [hit.document for hit in search_result]
                context = "\n\n".join(retrieve_context)
            except Exception as e:
                print(f"Context retrieval (with filter) failed for query '{query}': {e}")
                continue

            # Step 4: Generate answer with filter
            try:
                final_response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": answer_system_prompt},
                        {"role": "user", "content": f"Context:\n{context}\n\nPlease refer to this context to answer the following query."},
                        {"role": "user", "content": f"Query:\n{query}\n\nPlease provide a detailed, accurate response based on the context."}
                    ]
                )
                final_answer = final_response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Answer generation (with filter) failed for query '{query}': {e}")
                continue

            # Step 5: Retrieve context without filter
            try:
                search_result_without = qdrant_client.query(
                    collection_name=collectionName,
                    query_text=query,
                    limit=10,
                    query_filter=None
                )
                retrieve_context_without_filter = [hit.document for hit in search_result_without]
                context_without = "\n\n".join(retrieve_context_without_filter)
            except Exception as e:
                print(f"Context retrieval (without filter) failed for query '{query}': {e}")
                retrieve_context_without_filter = []
                context_without = ""

            # Step 6: Generate answer without filter
            try:
                final_response_without = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": answer_system_prompt},
                        {"role": "user", "content": f"Context:\n{context_without}\n\nPlease refer to this context to answer the following query."},
                        {"role": "user", "content": f"Query:\n{query}\n\nPlease provide a detailed, accurate response based on the context."}
                    ]
                )
                final_answer_without = final_response_without.choices[0].message.content.strip()
            except Exception as e:
                print(f"Answer generation (without filter) failed for query '{query}': {e}")
                final_answer_without = ""

            results.append({
                "query": query,
                "retrieved_context": retrieve_context,
                "retrieved_context_without_filter": retrieve_context_without_filter,
                "answer": final_answer,
                "answer_without_filter": final_answer_without,
                "expected_answer": expected_answer
            })
            print("***count**** ",count)
            count = count+1



    # Save results
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "query_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save results: {e}")
    
    return results


result = run_batch_query_pipeline("generated_qa_pairs.json")
print('****done*****')
