import os
import pandas as pd
import openai
from dotenv import load_dotenv
import evaluate

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Load dataset
data_path = "data/data.csv"
df = pd.read_csv(data_path)

# Initialize exact_match metric
exact_match = evaluate.load("exact_match")

def get_gpt_response(question):
    """Generates a response from GPT-4o for a given question."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are an AI assistant."},
                      {"role": "user", "content": question}]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""

# Generate GPT-4o responses and evaluate
predictions, references = [], []

for index, row in df.iterrows():
    question = row["Question"]
    actual_answer = row["Answer"]
    
    model_answer = get_gpt_response(question)

    predictions.append(model_answer)
    references.append(actual_answer)

    print(f"Q: {question}\nGPT-4o: {model_answer}\nActual: {actual_answer}\n")

# Compute Exact Match score
results = exact_match.compute(predictions=predictions, references=references)
print(f"Exact Match Score: {results['exact_match']:.2f}%")