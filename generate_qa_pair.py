import os
import json
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Initialize OpenAI API key (use your own key)
openai_client = OpenAI(api_key=os.getenv("api_key"))

# Folder containing patient record files
FOLDER_PATH = "mock_patient_records"
OUTPUT_JSON = "generated_qa_pairs.json"

# System prompt for GPT
system_prompt = (
    "You are a helpful assistant that generates 3 short question/answer pairs "
    "from a patient's medical record in structured JSON format."
)

# Function to generate QA pairs from a given text
def generate_qa(text):
    query = f"""Given the following patient record, generate 3 short question/answer pairs.

Return the output strictly in the following JSON format:
{{
  "qa_pairs": [
    {{"question": "Q1", "answer": "A1"}},
    {{"question": "Q2", "answer": "A2"}},
    {{"question": "Q3", "answer": "A3"}}
  ]
}}

Patient record:
{text}
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        json_response = response.choices[0].message.content.strip()
        # Parse the JSON response from GPT
        return json.loads(json_response)["qa_pairs"]
    except Exception as e:
        print(f"Error processing record: {e}")
        return None

# Main logic
def main():
    all_data = []

    for filename in tqdm(os.listdir(FOLDER_PATH)):
        if filename.endswith(".txt"):
            filepath = os.path.join(FOLDER_PATH, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()

            qa_pairs = generate_qa(content)
            if qa_pairs:
                all_data.append({
                    "filename": filename,
                    "qa_pairs": qa_pairs
                })

    # Save all generated QA pairs to a JSON file
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"✅ QA pairs saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
