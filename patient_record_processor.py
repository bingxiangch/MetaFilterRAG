import os
from docx import Document
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from openai import OpenAI
import re
# Load environment variables from the .env file
load_dotenv()

# Initialize OpenAI API key (use your own key)
openai_client = OpenAI(api_key=os.getenv("api_key"))

# Clients
qdrant_client = QdrantClient(
    url=os.getenv("qdrant_url"),
    api_key=os.getenv("qdrant_api_key") or None
)
collection_name = os.getenv("collection_name")



system_prompt = """
You are a clinical assistant AI helping build a patient record retrieval system. You will be given multiple paragraphs from a patient's record, and you must identify which section each paragraph belongs to.

Return only the name of the section for each paragraph, in the same order. No paragraph numbers, no extra words, no colons. One section per line.

The only valid sections are:
    "Basic Information",
    "Current Condition",
    "Treatment",
    "Medical History",
    "Lab Results",
    "Billing Information"

If the section is unclear, return "Unknown".
"""


# Function to read text from .docx
def read_docx_text(file_path):
    doc = Document(file_path)
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return paragraphs  # Skip the first paragraph

def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        # Split paragraphs by two newlines
        paragraphs = [para.strip() for para in content.split("\n\n") if para.strip()]
    return paragraphs


# Function to extract patient_id and department from filename
def extract_patient_info(filepath):
    basename = os.path.basename(filepath)
    name_part = basename.replace(".txt", "")
    
    # Split at the last underscore (in case name contains underscores)
    if "_" in name_part:
        name, department = name_part.rsplit("_", 1)
    else:
        name = name_part
        department = "general"
    
    # Replace underscores or normalize if needed
    patient_id = name.strip()
    return patient_id, department.strip()


# Function to classify sections for a file using GPT-3.5
def classify_sections_for_file(paragraphs):
    # Create the GPT-3 prompt by joining all paragraphs
    paragraphs_text = "\n".join(paragraphs)
    
    query = f"Paragraphs:\n{paragraphs_text}\n\nWhat section does each paragraph belong to? Please classify each paragraph into one of the following: Basic Information, Current Condition, Treatment, Medical History, Lab Results, Billing Information"
    
    # Send request to GPT-3.5
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    raw_lines = response.choices[0].message.content.strip().split("\n")

    # Clean each line to remove numbering, bullets, and whitespace
    clean_sections = []
    for line in raw_lines:
        # Remove "1.", "2)", "-", etc., at the beginning
        cleaned = re.sub(r"^\s*(\d+[\.\)]\s*|\-+)", "", line).strip()
        clean_sections.append(cleaned)

    return clean_sections


# --- Main Process ---
# Function to process each .docx file in the folder
def process_folder(folder_path):
    docs = []
    metadata = []
    ids = []
    
    # Initialize a global counter for IDs
    id_counter = 1
    
    # Iterate over all files in the directory
    for filename in os.listdir(folder_path):
        print('id_counter:',id_counter)
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # Read and process the document
            paragraphs = read_txt_file(file_path)
            patient_id, department = extract_patient_info(filename)

            # Call GPT-3.5 to classify sections for all paragraphs in the file
            sections = classify_sections_for_file(paragraphs)

            
            # # Ensure that the number of sections matches the number of paragraphs
            # if len(sections) != len(paragraphs):
            #     print(f"Warning: The number of sections returned does not match the number of paragraphs in file: {filename}")
            #     continue  # Skip this file if the section count doesn't match

            # Process each paragraph and its section
            for i, (paragraph, section) in enumerate(zip(paragraphs, sections)):
                docs.append(paragraph)
                metadata.append({
                    "patient_id": patient_id,
                    "department": department,
                    "section": section,
                    "source": f"Patient record: {patient_id.replace('_', ' ').title()}"
                })
                ids.append(id_counter)  # Assign the global ID
                id_counter += 1  # Increment the global counter

    # Now the ids will be sequential from 1 to n
    print(f"Docs length: {len(docs)}, Metadata length: {len(metadata)}, IDs length: {len(ids)}")

    # Add documents to Qdrant
    try:
        qdrant_client.add(
            collection_name="mock_patient_records",
            documents=docs,
            metadata=metadata,
            ids=ids
        )
        print(f"Successfully added {len(docs)} documents to Qdrant.")
    except Exception as e:
        print(f"An error occurred while adding documents to Qdrant: {e}")
    
    # Check the number of documents in Qdrant
    count = qdrant_client.count(collection_name="mock_patient_records")
    print(f"Documents in Qdrant collection: {count}")

# Specify your folder path containing .docx files
if __name__ == "__main__":
    folder = "./mock_patient_records"
    process_folder(folder)
