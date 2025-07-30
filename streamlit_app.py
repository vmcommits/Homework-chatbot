#Import various important libraries
import streamlit as st
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import tempfile

# --- CONFIG ---
DOC_ID = "1GeW5imtRw9AyOW0V9tiwxYsUGC8ZS6zXCr8GJLVqUwo"

# --- PAGE UI ---
st.set_page_config(page_title="Class IX Maths Homework Chatbot", layout="wide")
st.title("📚 Homework Helper Chatbot")
st.write("Ask anything from your notes. The chatbot only responds based on your document.")

# --- API Key ---
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.warning("Please enter your API key to start.", icon="⚠️")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# --- GOOGLE DOC AUTH (service account) ---
@st.cache_resource
def load_google_doc(doc_id):
    from google.oauth2.service_account import Credentials

    # Add your service account JSON credentials as a secret or upload file
    creds_dict = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/documents.readonly"])
    service = build('docs', 'v1', credentials=creds)
    document = service.documents().get(documentId=doc_id).execute()

    text = ""
    for content in document.get('body', {}).get('content', []):
        para = content.get("paragraph")
        if para:
            elements = para.get("elements", [])
            for e in elements:
                txt = e.get("textRun", {}).get("content", "")
                text += txt
    return text.strip()

# --- Load Notes & Embed ---
@st.cache_resource
def embed_notes(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, OpenAIEmbeddings())
    return vectorstore

# Load and process notes
with st.spinner("Loading notes..."):
    doc_text = load_google_doc(DOC_ID)
    vectorstore = embed_notes(doc_text)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask your homework question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    context = retriever.get_relevant_documents(prompt)
    context_text = "\n".join([doc.page_content for doc in context])

    system_prompt = f"You are a helpful AI tutor. Only answer based on the context provided.\n\nContext:\n{context_text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # Chat completion
    client = OpenAI(api_key=openai_api_key)
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
