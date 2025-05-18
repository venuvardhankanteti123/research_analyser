from flask import Flask, request, jsonify, send_from_directory
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI as genai
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

app = Flask(__name__, static_folder='static')

# Initialize LLM & Embeddings once
llm = genai(
    model="gemini-1.5-flash-001",
    google_api_key=os.getenv('key1'),
    temperature=0.7
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv('key1')
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# Globals to hold vectorstore & chain for current PDF
vectorstore = None
qa_chain = None

def build_vectorstore_and_chain(pdf_path):
    global vectorstore, qa_chain
    
    loader = PyPDFLoader(str(pdf_path))
    doc = loader.load()
    text = "\n".join(i.page_content for i in doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_text(text)

    # Build FAISS index (rebuild every upload)
    vectorstore = FAISS.from_texts(texts, embedding=embedding_model)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return text  # raw extracted text

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    file = request.files.get('pdf')
    if not file:
        return jsonify({"error": "No PDF uploaded"}), 400
    
    # Save temporarily in uploads folder
    os.makedirs('uploads', exist_ok=True)
    temp_path = os.path.join('uploads', file.filename)
    file.save(temp_path)
    
    raw_text = build_vectorstore_and_chain(temp_path)
    
    return jsonify({"raw_text": raw_text})

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    raw_text = data.get('raw_text')
    if not raw_text:
        return jsonify({"error": "No text to summarize"}), 400

    prompt = f"Summarize the following research paper text briefly:\n\n{raw_text}"

    summary = llm(prompt)
    if isinstance(summary, dict) and "text" in summary:
        summary_text = summary["text"]
    else:
        summary_text = summary if isinstance(summary, str) else str(summary)

    return jsonify({"summary": summary_text})
@app.route('/ask', methods=['POST'])
def ask():
    global qa_chain
    if not qa_chain:
        return jsonify({"error": "No document loaded yet"}), 400

    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    response = qa_chain.invoke({"question": question})
    
    return jsonify({
        "answer": response.get("answer"),
        "sources": [doc.page_content for doc in response.get("source_documents", [])]
    })

if __name__ == '__main__':
    app.run(debug=True)
