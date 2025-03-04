import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import evaluate

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure API key is loaded
if not OPENAI_API_KEY:
    raise ValueError("API Key not found. Make sure to set OPENAI_API_KEY in the .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load dataset
data_path = "data/data.csv"
df = pd.read_csv(data_path)

# Ensure column names match
expected_columns = ["Question", "Answer"]
if not all(col in df.columns for col in expected_columns):
    raise ValueError(f"CSV file is missing expected columns: {expected_columns}")

# Initialize exact_match metric
exact_match = evaluate.load("exact_match")

# Create results directory
results_dir = "result"
os.makedirs(results_dir, exist_ok=True)
result_file = os.path.join(results_dir, "results.csv")

def get_gpt_response(question):
    """Generates a response from GPT-4o for a given question."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer these questions from Masoyinbo, a YouTube show where guests respond to questions in Yoruba. Keep your answers brief and direct, no extra details, just straight to the point."},
                {"role": "user", "content": question},
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""

# Generate GPT-4o responses and evaluate
results = []

for _, row in df.iterrows():
    question = row["Question"]
    actual_answer = row["Answer"]

    model_answer = get_gpt_response(question)
    score = 1 if str(model_answer).strip().lower() == str(actual_answer).strip().lower() else 0

    results.append([question, model_answer, actual_answer, score])

# Save results to CSV
results_df = pd.DataFrame(results, columns=["Question", "GPT-4o_Answer", "Actual_Answer", "Score"])
results_df.to_csv(result_file, index=False)

print(f"Results saved in {result_file}")
