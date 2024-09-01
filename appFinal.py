from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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

def get_significant_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words

def summarize_input(user_input):
    words = get_significant_words(user_input)
    most_common_words = [word for word, count in Counter(words).most_common(3)]
    return most_common_words

def filter_relevant_sentences(pdf_text, keywords):
    sentences = sent_tokenize(pdf_text)
    relevant_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    return relevant_sentences

def summarize_text(text, word_limit=400):
    sentences = sent_tokenize(text)
    summary = ''
    for sentence in sentences:
        if len(summary.split()) + len(sentence.split()) <= word_limit:
            summary += ' ' + sentence
        else:
            break
    return summary

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

    # Use retrieved documents to build context for response combination
    pdf_text = ' '.join([doc.page_content for doc in docs])
    
    # Summarize the user input to get key phrases
    keywords = summarize_input(user_input)
    
    # Filter relevant sentences based on keywords
    relevant_sentences = filter_relevant_sentences(pdf_text, keywords)
    
    # Summarize relevant sentences if too long
    filtered_text = ' '.join(relevant_sentences)
    summarized_text = summarize_text(filtered_text)

    # Log the context before invoking LLM
    print("Context for LLM:", summarized_text)

    # Use the summarized PDF content and user input to get LLM response
    llm_query = f"Context: {summarized_text}\n\nQuestion: {user_input}"
    llm_response = llm.invoke(llm_query)

    # Return the combined response
    return jsonify({
        'pdf_content': summarized_text,
        'llm_response': llm_response
    })

if __name__ == '__main__':
    app.run(debug=True)




