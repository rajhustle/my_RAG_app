import logging
from flask import Flask, request, jsonify, render_template, redirect, url_for
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models import ChatOllama

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s:')
app = Flask(__name__)

# Load and split the PDF
def load_and_split_pdf(file_path):
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    return splitter.split_documents(data)

# Embed the chunks to vector DB
def embed_chunks(chunks):
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    db = Chroma(collection_name='local-rag', persist_directory='chroma', embedding_function=embeddings)
    db.add_documents(chunks)
    db.persist()
    return db

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/embed", methods=["POST"])
def upload_embed():
    try:
        file = request.files['file']
        file_path = "./" + file.filename
        file.save(file_path)
        chunks = load_and_split_pdf(file_path)
        embed_chunks(chunks)
        msg = f"File '{file.filename}' uploaded and embedded successfully."
        return render_template("home.html", message=msg)
    except Exception as e:
        err = f"ERROR during embedding: {str(e)}"
        return render_template("home.html", error=err)

@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "POST":
        question = request.form["query"]
        try:
            db = Chroma(collection_name='local-rag', persist_directory='chroma', embedding_function=OllamaEmbeddings(model='nomic-embed-text'))
            model = ChatOllama(model='mistral')
            relevant_docs = db.similarity_search(question)
            context = " ".join([doc.page_content for doc in relevant_docs])
            answer = model.invoke("Answer this question based only on this context:\\n" + context + "\\nQuestion: " + question)
            return render_template("query.html", answer=answer, query=question)
        except Exception as e:
            return render_template("query.html", error=f"ERROR: {str(e)}", query=question)
    return render_template("query.html")

if __name__ == "__main__":
    app.run(port=8080, debug=True)

import logging

# This sets up basic logging so messages will show up in your terminal.
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s:')

from flask import Flask, request, jsonify

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models import ChatOllama

app = Flask(__name__)

# Load and split the PDF
def load_and_split_pdf(file_path):
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    return splitter.split_documents(data)

# Embed the chunks to vector DB
def embed_chunks(chunks):
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    db = Chroma(collection_name='local-rag', persist_directory='chroma', embedding_function=embeddings)
    db.add_documents(chunks)
    db.persist()
    return db

# PDF upload and embedding route with error logging
@app.route('/embed', methods=['POST'])
def upload_embed():
    try:
        file = request.files['file']
        file_path = "./" + file.filename
        print(f"Saving uploaded file to {file_path}...")
        file.save(file_path)
        print("File saved. Loading and splitting PDF...")
        chunks = load_and_split_pdf(file_path)
        print(f"PDF loaded and split into {len(chunks)} chunks. Embedding...")
        embed_chunks(chunks)
        print("PDF embedding complete.")
        return jsonify({"message": "File embedded successfully"}), 200
    except Exception as e:
        print("ERROR during embedding:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Query route
@app.route('/query', methods=['POST'])
def query_pdf():
    question = request.json['query']
    db = Chroma(collection_name='local-rag', persist_directory='chroma', embedding_function=OllamaEmbeddings(model='nomic-embed-text'))
    model = ChatOllama(model='mistral')
    relevant_docs = db.similarity_search(question)
    context = " ".join([doc.page_content for doc in relevant_docs])
    answer = model.invoke("Answer this question based only on this context:\\n" + context + "\\nQuestion: " + question)
    return jsonify({"answer": answer}), 200

if __name__ == "__main__":
    logging.info("Something happened!")
    app.run(port=8080, debug=True)




    
