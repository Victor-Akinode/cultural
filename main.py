import os
import pandas as pd
import openai
from evaluate import load
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Ensure column names match your dataset
questions = df["Questions"].tolist()
answers = df["Answer"].tolist()

# Function to get GPT-4o response
def get_gpt_response(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                      {"role": "user", "content": question}]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""

# Get predictions from GPT-4o
predictions = [get_gpt_response(q) for q in questions]

# Load exact_match metric
exact_match = load("exact_match")

# Compute accuracy
results = exact_match.compute(predictions=predictions, references=answers)
print(f"Exact Match Score: {results['exact_match']:.2f}%")