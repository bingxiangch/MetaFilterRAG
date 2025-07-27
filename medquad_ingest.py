import os
import xml.etree.ElementTree as ET
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams, CollectionStatus
from dotenv import load_dotenv
import hashlib
from sentence_transformers import SentenceTransformer


# Load the SentenceTransformer model (384-dim)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load environment variables from .env
load_dotenv()

# Qdrant setup
qdrant_client = QdrantClient(
    url=os.getenv("qdrant_url"),
    api_key=os.getenv("qdrant_api_key") or None,
)

collection_name = os.getenv("collection_name", "medical_qa")

# Create collection if it doesn’t exist
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )


def parse_xml_file(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    doc_id = root.attrib.get("id")
    source = root.attrib.get("source")
    url = root.attrib.get("url")
    focus = root.findtext("Focus", default="")

    qa_pairs = []
    for pair in root.findall(".//QAPair"):
        question = pair.findtext("Question")
        answer = pair.findtext("Answer")
        qtype = pair.find("Question").attrib.get("qtype", "")

        if question and answer:
            qa_pairs.append({
                "question": question.strip(),
                "answer": answer.strip(),
                "qtype": qtype,
                "doc_id": doc_id,
                "source": source,
                "url": url,
                "focus": focus
            })

    return qa_pairs


def embed_text(text):
    embedding = embedding_model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

def process_xml_folder_recursive(root_folder, batch_size=100):
    id_counter = 1
    all_points = []

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.endswith(".xml"):
                continue

            file_path = os.path.join(dirpath, filename)
            qa_entries = parse_xml_file(file_path)

            for qa in qa_entries:
                vector = embed_text(qa['question'])

                point = PointStruct(
                    id=id_counter,
                    vector=vector,
                    payload={
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "qtype": qa["qtype"],
                        "doc_id": qa["doc_id"],
                        "focus": qa["focus"],
                        "source": qa["source"],
                        "url": qa["url"],
                    }
                )
                all_points.append(point)
                id_counter += 1

                # Upload in batches
                if len(all_points) >= batch_size:
                    print(f"Inserting batch of {len(all_points)} points...")
                    qdrant_client.upsert(collection_name=collection_name, points=all_points)
                    all_points = []  # Clear buffer

    # Insert remaining points
    if all_points:
        print(f"Inserting final batch of {len(all_points)} points...")
        qdrant_client.upsert(collection_name=collection_name, points=all_points)

    print("All data inserted into Qdrant.")



if __name__ == "__main__":
    root_folder = "./MedQuAD"  # Root folder containing all subfolders like 1_CancerGov_QA, 2_GARD_QA
    process_xml_folder_recursive(root_folder)