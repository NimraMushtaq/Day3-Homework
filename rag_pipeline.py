import json
import os
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from typing import List
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_api_key)

# Load text file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return content

# Split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Specify the path to the text file
text_file_path = "docs/intro-to-llms-karpathy.txt"
file_text = load_text_file(text_file_path)

# Create chunks of text
chunked_text = get_text_chunks(file_text)

# Define a custom embedding function using Gemini API
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = "models/embedding-001"
        title = "Custom query"
        try:
            embedding_response = genai.embed_content(
                model=model,
                content=input,
                task_type="retrieval_document",
                title=title
            )
            return embedding_response["embedding"]
        except Exception as e:
            print(f"Embedding failed for input: {input}. Error: {e}")
            return []

# Create directory for database if it doesn't exist
db_folder = "chroma_db"
if not os.path.exists(db_folder):
    os.makedirs(db_folder)

# Create ChromaDB instance
def create_chroma_db(documents: List[str], path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    try:
        db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        print(f"Loaded existing collection '{name}'.")
    except Exception as e:
        print(f"Collection '{name}' not found. Creating a new collection.")
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        for i, d in enumerate(documents):
            try:
                db.add(documents=[d], ids=[str(i)])
                if (i + 1) % 1000 == 0:
                    print(f"Added {i + 1} chunks to the collection.")
            except Exception as add_e:
                print(f"Failed to add chunk {i}: {add_e}")
    return db, name

# Specify the path and collection name for ChromaDB
db_name = "rag_experiment"
db_path = os.path.join(os.getcwd(), db_folder)
db, db_name = create_chroma_db(chunked_text, db_path, db_name)

# Load an existing Chroma collection
def load_chroma_collection(path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

# Load the database
db = load_chroma_collection(db_path, db_name)

def generate_answer(prompt: str):
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        result = model.generate_content(prompt)
        print("Result Json : ", result)
        answer_text = result.candidates[0].content.parts[0].text
        return answer_text
    except Exception as e:
        print(f"Answer generation failed: {e}")
        return ""

def get_relevant_passage(query: str, db, n_results: int = 1):
    try:
        results = db.query(query_texts=[query], n_results=n_results)
        if 'documents' in results and len(results['documents']) > 0:
            return results['documents'][0]
        else:
            print("No relevant passage found.")
            return None
    except Exception as e:
        print(f"Query failed: {e}")
        return None

# Generate a prompt for the RAG model
def make_rag_prompt(query: str, relevant_passage: str):
    if isinstance(relevant_passage, list):
        relevant_passage = relevant_passage[0] if relevant_passage else ""
    # Ensure relevant_passage is a string
    escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
strike a friendly and conversational tone.
QUESTION: '{query}'
PASSAGE: '{escaped_passage}'

ANSWER:
"""
    return prompt

# Process a query, generate an answer, and save output to a JSON file
def qa_chain(query: str, output_file: str):
    relevant_text = get_relevant_passage(query, db, n_results=1)
    if relevant_text:
        final_prompt = make_rag_prompt(query, relevant_text)
        answer = generate_answer(final_prompt)
        output_data = {
            "question": query,
            "answer": answer,
            "context": relevant_text
        }
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, indent=4, ensure_ascii=False)
        print(f"Result saved to {output_file}")
        return output_data
    else:
        return {"result": "", "source_documents": []}

# Example usage
query_string = "What methods does the language model use for data collection and organization when responding to queries?"
qa_chain(query_string, 'output.json')