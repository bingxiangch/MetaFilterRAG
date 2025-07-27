import os
import xml.etree.ElementTree as ET
import random
import json

# Configuration
medquad_dir = "MedQuAD"
output_file = "medquad_qa_subset_500.json"
sample_size = 500

all_qa_pairs = []

# Step 1: Walk through all subfolders
for root, dirs, files in os.walk(medquad_dir):
    for file in files:
        if file.endswith(".xml"):
            file_path = os.path.join(root, file)
            try:
                tree = ET.parse(file_path)
                root_elem = tree.getroot()
                qapairs = root_elem.find("QAPairs")
                if qapairs is not None:
                    for pair in qapairs.findall("QAPair"):
                        question = pair.find("Question")
                        answer = pair.find("Answer")
                        qtype = question.attrib.get("qtype") if question is not None else None

                if question is not None and question.text and answer is not None and answer.text:
                    all_qa_pairs.append({
                        "question": question.text.strip(),
                        "answer": answer.text.strip(),
                        "qtype": qtype
                    })

            except ET.ParseError as e:
                print(f"Skipping {file_path}: Parse error {e}")

# Step 2: Random sampling
print(f"Total QA pairs found: {len(all_qa_pairs)}")
sampled = random.sample(all_qa_pairs, min(sample_size, len(all_qa_pairs)))

# Step 3: Save to file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(sampled, f, indent=2, ensure_ascii=False)

print(f"Saved {len(sampled)} QA pairs to {output_file}")
