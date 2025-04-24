import pandas as pd
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=API_KEY)

def get_gpt4o_answer(question, options):
    try:
        prompt = (
            "As a subject matter expert in Yoruba Language, analyze and determine the best answer "
            "to this question accurately and concisely. There are five options, "
            f"Options: {options[0]}, {options[1]}, {options[2]}, {options[3]}, {options[4]}. "
            "Respond strictly with only the correct option (e.g., 'option_1', 'option_2', etc.)."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI that answers Yoruba language questions accurately."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip().rstrip(".")
    except Exception as e:
        return f"Error: {str(e)}"

def update_answers(csv_file='refined_data.csv', output_file='result/final_result.csv'):
    # Load the CSV file
    df = pd.read_csv(csv_file).fillna("")

    # Create result folder if it doesn't exist
    os.makedirs("result", exist_ok=True)
    
    # Create new columns
    df['GPT4o_Answer'] = ''  # GPT-4o generated answer
    df['GPT_Match'] = 0      # 1 if match, else 0
    
    # Iterate and evaluate
    correct_count = 0
    total_questions = len(df)
    
    for index, row in df.iterrows():
        question = str(row['Question'])
        answer_value = str(row['Answer']).strip().lower()
        options = [str(row[f'option_{i}']) for i in range(1, 6)]
        
        gpt_answer = get_gpt4o_answer(question, options).strip().lower()
        gpt_answer_clean = gpt_answer.rstrip(".")
        
        df.at[index, 'GPT4o_Answer'] = gpt_answer_clean

        if gpt_answer_clean == answer_value:
            df.at[index, 'GPT_Match'] = 1
            correct_count += 1

    # Calculate accuracy
    accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    df['Score (%)'] = accuracy  # Add accuracy score for each row for visibility

    # Save to file
    df.to_csv(output_file, index=False)
    print(f"Result saved at {output_file}")
    print(f"GPT-4o Accuracy: {accuracy:.2f}%")

# Run the function
update_answers()