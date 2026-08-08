import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Page setup
st.set_page_config(page_title="Notebook QA Bot", layout="wide")
st.title("📚 School Notebook QA Assistant")
st.write("Upload your notebook PDF and ask questions strictly based on its contents.")

# Sidebar for API Key & File Upload
with st.sidebar:
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    pdf_docs = st.file_uploader("Upload Notebook (PDF)", accept_multiple_files=True, type=["pdf"])

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(api_key):
    # STRICT PROMPT: Internet or outside knowledge is strictly forbidden
    prompt_template = """
    You are a helpful study assistant for school students. 
    Answer the question ONLY based on the provided context extracted from the uploaded notebook.
    
    Strict Rules:
    1. If the answer is NOT present in the provided context, respond EXACTLY: "Sorry, this topic is not mentioned in your uploaded notebook."
    2. Do NOT use any external knowledge or internet information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# User input question
user_question = st.text_input("Ask a question from your uploaded notebook:")

if pdf_docs and api_key:
    if st.button("Process Notebook"):
        with st.spinner("Processing PDF..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            st.session_state.vector_store = get_vector_store(text_chunks, api_key)
            st.success("Notebook processed successfully! Now ask your questions.")

if user_question:
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif "vector_store" not in st.session_state:
        st.error("Please upload and process a notebook first.")
    else:
        docs = st.session_state.vector_store.similarity_search(user_question)
        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("### Answer:")
        st.write(response["output_text"])
