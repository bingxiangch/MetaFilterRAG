import openai
from faker import Faker
import random
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI API key (use your own key)
openai_client = OpenAI(api_key=os.getenv("api_key"))

# Setup Faker for names and random data
fake = Faker()
departments = ["Cardiology", "Neurology", "Oncology"]

# List of 100 unique first and last names
first_names = ["Juhani", "Monica", "James", "Barbara", "Rachel", "Alex", "John", "Sophia", "Liam", "Olivia",
               "Emma", "Noah", "Ava", "William", "Mason", "Isabella", "Ethan", "Abigail", "Michael", "Lucas",
               "Charlotte", "Amelia", "Benjamin", "Ella", "Henry", "Scarlett", "Jack", "Mia", "Daniel", "Harper",
               "Matthew", "Zoe", "David", "Aria", "Joseph", "Grayson", "Lily", "Samuel", "Madison", "Caleb",
               "Chloe", "Owen", "Violet", "Nathan", "Ruby", "Leo", "Hazel", "Wyatt", "Leah", "Isaac", "Addison",
               "Eleanor", "Aiden", "Carter", "Evelyn", "Jackson", "Landon", "Gabriel", "Avery", "Elijah", "Sophie",
               "Jack", "Zachary", "Grace", "Sebastian", "Lillian", "Dylan", "Hannah", "Aaron", "Lily", "Mila",
               "Joshua", "Victoria", "Ryan", "Brooklyn", "Sebastian", "Megan", "Ella", "Eli", "Andrew", "Maya",
               "Jaxon", "Daniel", "Ella", "Lillian", "Mason", "Santiago", "Amaya", "Madeline", "Layla", "Michael"]

last_names = ["Virtanen", "Peterson", "Young", "Cantu", "Smith", "Johnson", "Doe", "Brown", "Davis", "Miller",
              "Williams", "Garcia", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
              "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez",
              "Clark", "Ramirez", "Lewis", "Roberts", "Walker", "Young", "King", "Wright", "Scott", "Torres",
              "Nguyen", "Hill", "Adams", "Baker", "Nelson", "Carter", "Mitchell", "Perez", "Evans", "Green",
              "Turner", "Collins", "Murphy", "Cook", "Rivera", "Morris", "Bell", "Reyes", "Gomez", "Sanders",
              "Price", "Wood", "Cooper", "Riley", "Howard", "Ward", "Flores", "King", "Hughes", "Simmons", "Foster"]

# System prompt for generating patient records
system_prompt = (
    "You are a medical assistant generating detailed and realistic mock patient records for clinical documentation. "
    "Each record should include the following sections, each as a short paragraph in this order: "
    "Use the following example as a style and structure reference, where each section is formatted in its own paragraph:\n\n"

    "Basic Information: [Full Name] is a [Age]-year-old [Gender] living in [City], Finland. Finnish is their native language. "
    "They are a [Profession] who lives with [Family] in a [Home Type]. They enjoy [Hobby] as a hobby and frequently [Activity].\n\n"

    "Current Condition: At a recent routine check-up, [Full Name] reported [Condition]. "
    "They have [Symptom] and feel [Fatigue Level] by late afternoon. Their mobility remains [Mobility], and they stay socially active through [Social Activity].\n\n"

    "Treatment: [Full Name] uses a daily [Medication] for [Condition] and takes [Medication] for [Cholesterol/Other] and [Medication] for cardiovascular protection. "
    "They occasionally use [Emergency Medication] during flare-ups. They follow a [Diet] diet and checks their [Health Metric] regularly.\n\n"

    "Medical History: They were diagnosed with [Condition Diagnosed] [Years Ago]. They have [Other Conditions] and experienced a [Medical Event] [Years Ago]. "
    "No known drug allergies. Family history of [Family History].\n\n"

    "Lab Results: [Lab Test 1] was [Lab Test Value 1], slightly above the target. [Lab Test 2] showed [Lab Test Value 2]. "
    "[Blood Pressure] measured at [Blood Pressure Value]. Oxygen saturation at rest was [Oxygen Saturation]. Results indicate [Condition Status].\n\n"

    "Billing Information: Total bill: [Bill Amount]. Insurance covered [Insurance Coverage]%. Patient copay: [Patient Copay]. Invoice issued on [Invoice Date].\n\n"

    "Create a unique and realistic record for each patient based on this format. Ensure all details are logically consistent with age, condition, and department. "
    "Each section should be presented as a short paragraph, with no extra line breaks or gaps between sections."
)

# Create output folder
output_dir = "mock_patient_records"
os.makedirs(output_dir, exist_ok=True)

# Generate 100 unique patient records
for i in range(100):
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    full_name = f"{first_name} {last_name}"
    department = random.choice(departments)

    user_query = f"Create a mock patient record for {full_name} in the {department} department."

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )

        content = response.choices[0].message.content.strip()

        # Split content into sections (6 paragraphs)
        paragraphs = content.split("\n\n")
        if len(paragraphs) != 6:
            print(f"Unexpected number of paragraphs for {full_name}, skipping...")
            continue

        # Save to TXT file
        filename = os.path.join(output_dir, f"{full_name.replace(' ', ' ')}_{department}.txt")
        with open(filename, 'w') as file:
            for paragraph in paragraphs:
                file.write(paragraph + "\n\n")

        print(f"Saved: {filename}")
        time.sleep(1)  # To avoid hitting rate limits

    except Exception as e:
        print(f"Failed to generate record for {full_name}: {str(e)}")
