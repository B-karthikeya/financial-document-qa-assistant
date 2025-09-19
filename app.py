import streamlit as st
import pandas as pd
import pdfplumber
import tempfile, os, io, json, requests
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"  # change to your local model name

st.set_page_config(page_title="Financial Document Q&A", layout="wide")
st.title("📊 Financial Document Q&A Assistant")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "X" not in st.session_state:
    st.session_state.X = None

# ---------- Document Extraction ----------
def extract_from_pdf_bytes(b):
    pages = []
    bio = io.BytesIO(b)
    with pdfplumber.open(bio) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages

def extract_from_excel_bytes(b):
    bio = io.BytesIO(b)
    try:
        xl = pd.read_excel(bio, sheet_name=None)
        return xl
    except Exception as e:
        st.warning(f"Excel read error: {e}")
        return {}

def split_text_to_chunks(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_index(chunks):
    vec = TfidfVectorizer().fit(chunks)
    X = vec.transform(chunks)
    return vec, X

def retrieve_top_k(question, vec, X, chunks, k=3):
    qv = vec.transform([question])
    scores = (X * qv.T).toarray().ravel()
    idxs = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in idxs if scores[i] > 0]

def call_ollama(prompt, model=MODEL_NAME, timeout=60):
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        js = r.json()
        return js.get("response", r.text)
    except Exception as e:
        return f"[Error calling Ollama API: {e}]"

# ---------- Streamlit UI ----------
uploaded = st.file_uploader("Upload PDF or Excel file", type=['pdf','xls','xlsx'])

if uploaded:
    with st.spinner("Extracting..."):
        text_data = ""
        if uploaded.name.lower().endswith(".pdf"):
            pages = extract_from_pdf_bytes(uploaded.read())
            text_data = "\n\n".join([p[1] for p in pages])
        else:
            xl = extract_from_excel_bytes(uploaded.read())
            sheet_texts = []
            for sname, df in xl.items():
                sheet_texts.append(f"Sheet: {sname}\n{df.to_csv(index=False)}")
            text_data = "\n\n".join(sheet_texts)

        chunks = split_text_to_chunks(text_data, chunk_size=400, overlap=50)
        st.session_state.chunks = chunks
        if chunks:
            vec, X = build_index(chunks)
            st.session_state.vectorizer = vec
            st.session_state.X = X
            st.success(f"✅ Extracted {len(chunks)} chunks from {uploaded.name}")

st.subheader("Ask a question about your document")
question = st.chat_input("Type your question here...")
if question:
    if not st.session_state.chunks:
        st.warning("Please upload a document first.")
    else:
        top_chunks = retrieve_top_k(question, st.session_state.vectorizer, st.session_state.X, st.session_state.chunks, k=4)
        context = "\n\n".join(top_chunks)
        prompt = f"""You are a financial assistant. Use the following context to answer.
If the answer is not present, reply 'Not found in the document.'

Context:
{context}

Question: {question}
Answer:"""
        answer = call_ollama(prompt)
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)
