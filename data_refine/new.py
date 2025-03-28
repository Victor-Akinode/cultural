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
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def update_answers(csv_file='refined_updated_data.csv', output_file='final_result.csv'):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Create new columns
    df['GPT4o_Answer'] = ''  # GPT-4o generated answer (in option_ format)
    df['GPT_Match'] = 0  # 1 if GPT-4o's answer matches the original Answer, else 0
    
    # Iterate over the rows
    correct_count = 0
    total_questions = len(df)
    
    for index, row in df.iterrows():
        answer_value = row['Answer']
        
        # Get GPT-4o answer
        options = [row['option_1'], row['option_2'], row['option_3'], row['option_4'], row['option_5']]
        gpt_answer = get_gpt4o_answer(row['Question'], options)
        df.at[index, 'GPT4o_Answer'] = gpt_answer
        
        # Evaluate GPT-4o correctness
        if gpt_answer.strip().lower() == answer_value.strip().lower():
            df.at[index, 'GPT_Match'] = 1
            correct_count += 1
    
    # Calculate GPT-4o accuracy
    accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Updated CSV file saved as {output_file}")
    print(f"GPT-4o Accuracy: {accuracy:.2f}%")

# Example usage
update_answers()
