import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def web_search(query):
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return tavily.search(query=query, search_depth="advanced")["results"]

st.set_page_config(page_title="Khalid's RAG & LLM Hub", page_icon="📚", layout="wide")
st.title("📚 Khalid's RAG & LLM Document Intelligence Hub")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    process_button = st.button("Process Documents")

if process_button and uploaded_files:
    if not os.getenv("GOOGLE_API_KEY"):
        st.sidebar.error("Please add your GOOGLE_API_KEY to the .env file!")
    else:
        with st.spinner("Processing Documents and Creating Vector Database..."):
            raw_text = get_pdf_text(uploaded_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.sidebar.success(f"Success! Created {len(text_chunks)} chunks and saved to FAISS DB.")

st.subheader("Chat with your Documents\nBy/Khalid Mabrouk")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    context = ""
    if os.path.exists("faiss_index"):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_query)
        context = "\n".join([d.page_content for d in docs])

    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:-1]])

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
    
    initial_prompt = f"System: Provide a detailed, comprehensive, and well-structured answer strictly using the provided context. If the answer is not in the context, output exactly and only the word 'SEARCH_WEB'.\n\nChat History:\n{history_text}\n\nContext: {context}\n\nQuestion: {user_query}"
    
    try:
        response = llm.invoke(initial_prompt)
        final_answer = ""
        
        if "SEARCH_WEB" in response.content:
            st.warning("Information not found in PDFs. Searching the web...")
            web_results = web_search(user_query)
            web_context = "\n".join([r['content'] for r in web_results])
            
            final_prompt = f"System: Provide a detailed, comprehensive, and well-structured answer using the following web search results.\n\nChat History:\n{history_text}\n\nWeb Data: {web_context}\n\nQuestion: {user_query}"
            final_response = llm.invoke(final_prompt)
            final_answer = final_response.content
        else:
            final_answer = response.content
            
        with st.chat_message("assistant"):
            st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
    except Exception as e:
        st.error(f"An error occurred: {e}. Please check your API keys and internet connection.")