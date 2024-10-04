import json
import os
from rag_pipeline import qa_chain
import json

# Function to process all questions and generate answers
def process_all_questions(input_json_file: str, output_json_file: str):
    # Load the input JSON file containing the list of questions
    with open(input_json_file, 'r', encoding='utf-8') as json_file:
        questions_list = json.load(json_file)

    # Initialize an empty list to store the final results
    results = []

    try:
        # Iterate over each question in the input JSON
        for question_data in questions_list:
            question = question_data["question"]

            # Call the qa_chain function to get the answer and context
            qa_chain(question, 'temp.json')

            # Read from the 'output.json' where qa_chain stores the result for each query
            with open('output.json', 'r', encoding='utf-8') as temp_file:
                result_data = json.load(temp_file)

            # Append the question, answer, and context to the results list
            results.append({
                "question": result_data["question"],
                "answer": result_data["answer"],
                "contexts": result_data["context"]
            })

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Save the results to the final output JSON file
        with open(output_json_file, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, indent=4, ensure_ascii=False)

        print(f"All questions processed and saved to {output_json_file}")


# Example usage
input_json_file = 'questions.json'
output_json_file = 'my_rag_output.json'

# Process all questions and generate answers
process_all_questions(input_json_file, output_json_file)

