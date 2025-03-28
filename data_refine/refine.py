import pandas as pd

def update_answers(csv_file='updated_data.csv', output_file='refined_updated_data.csv'):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Iterate over the rows
    for index, row in df.iterrows():
        answer_value = row['Answer']
        
        # Check if Answer matches any of the option columns
        for option in ['option_1', 'option_2', 'option_3', 'option_4', 'option_5']:
            if row[option] == answer_value:
                df.at[index, 'Answer'] = option
                break  # Stop checking after the first match
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Updated CSV file saved as {output_file}")

# Example usage
update_answers()
