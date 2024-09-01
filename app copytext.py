from flask import Flask, render_template, request, jsonify
import os
from flask_cors import CORS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import fitz  # PyMuPDF



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Huggingface model
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")

# Load documents and create embeddings
loader = TextLoader('Sample.txt')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']
    docs = retriever.get_relevant_documents(user_input)
    completion = llm.invoke(user_input)
    return jsonify({'answer': completion})

if __name__ == '__main__':
    app.run(debug=True)

