from flask import Flask, render_template, request, jsonify
import os
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import fitz  # PyMuPDF

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Huggingface model
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ''
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def create_document_objects(text_chunks):
    return [Document(page_content=chunk) for chunk in text_chunks]

# Path to the PDF file in the same folder as app.py
pdf_path = 'book1-for-CD.pdf'
pdf_text = extract_text_from_pdf(pdf_path)

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_text(pdf_text)

# Create Document objects
documents = create_document_objects(texts)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(documents, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']
    docs = retriever.get_relevant_documents(user_input)
    
    # Use retrieved documents to influence the language model response
    pdf_content = ' '.join([doc.page_content for doc in docs])
    
    # Truncate pdf_content to avoid exceeding token limits
    max_token_length = 32000  # Define a maximum length that ensures the input is within limits
    if len(pdf_content) > max_token_length:
        pdf_content = pdf_content[:max_token_length]  # Truncate if necessary

    completion = llm.invoke(f"{pdf_content}\n\nQuestion: {user_input}")
    
    return jsonify({'answer': completion})

if __name__ == '__main__':
    app.run(debug=True)


