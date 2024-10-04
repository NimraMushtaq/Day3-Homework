# AI Bootcamp in India
## Homework 3 : Building a Question Answering RAG System using Gemini

### Overview

As part of Homework 3 in the AI Bootcamp, I implemented a Question Answering (QA) system using the Retrieval-Augmented Generation (RAG) technique. 
The task was to build a system that answers questions based on the transcript of Andrey Karpathy’s talk, "[[1hr Talk] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)."
The transcript is included in this repository at [docs/intro-to-llms-karpathy.txt](docs/intro-to-llms-karpathy.txt).
In this project, I utilized the Google Gemini API for the RAG implementation.




### 1. Pre-requisites
In order to complete the homework assignment you will need to have
- Python 3.10+
- Gemini API Key


### 2. Setup
- Fork this repository from GitHub
- Install dependencies by running
```sh
pip install -r requirements.txt
```
- Rename .env-example to .env
- Set your `GOOGLE_API_KEY` in the .env file
```sh
GOOGLE_API_KEY=sk-...
```

### 3. Implemenation

The goal was to create a Naive RAG pipeline to answer questions based on the provided document transcript.
To implement a Question Answering system using a Naive RAG approach, I performed the following steps:

- **Data Ingestion**:
  - Split the document into chunks.
  - Each chunk was embedded using Google Gemini’s embedding model.
  - The embeddings were stored in a vector database ChromaDB.
- **RAG**:
  - Embed the question.
  - Retrieve the relevant context from the vector database.
  - Prompt the LLM (Gemini) with the question and retrieved context to generate an answer.

### 4. Generating Answers to the Question List

To fulfill the assignment requirements, I wrote a script that generates answers for a list of 50 predefined questions. 
The results were saved in the required JSON format. Here's a snippet of the script:

```python
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
```

### 5. Evaluation

The original repository provided an eval.py script to calculate the RAGAS metrics and generate a ragas_score. 
Unfortunately I was unable to run the eval.py successfully on my Gemini-based implementation.

```sh
python eval.py path-to-your-output.json
```

Although I couldn't calculate the evaluation score, I am still submitting the full pipeline implementation along with the generated answers in the my_rag_output.json file. 
These answers were generated using Google Gemini.





