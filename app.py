import os
import streamlit as st
from dotenv import load_dotenv
import pdfplumber

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# UI config
st.set_page_config(page_title="Legal Case Analyzer", layout="wide")
st.title("⚖️ Legal Document Analyzer & Case Summary Generator")

# Validate API Key
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Prompt template
prompt_template = """
You are a legal assistant. Use the following context to answer the legal question clearly and concisely.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# PDF Upload
uploaded_file = st.file_uploader("📄 Upload a Legal Case PDF", type="pdf")

if uploaded_file:
    st.success("📄 PDF uploaded successfully")

    # Avoid reprocessing same PDF
    if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
        # Extract text
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        st.write("📝 Extracted text length:", len(text))

        if len(text.strip()) == 0:
            st.warning("⚠️ Could not extract text. Please upload a different PDF.")
            st.stop()

        if len(text) > 20000:
            st.warning("⚠️ PDF is too large. Please upload a shorter document.")
            st.stop()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.create_documents([text])
        st.write("📚 Number of chunks created:", len(chunks))

        # Limit chunks
        if len(chunks) > 200:
            chunks = chunks[:200]
            st.warning("⚠️ Only first 200 chunks are used to avoid memory issues.")

        # Store in session
        st.session_state.last_file = uploaded_file.name
        st.session_state.chunks = chunks
        st.session_state.faiss_db = FAISS.from_documents(
            chunks,
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.faiss_db.as_retriever(),
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )
        st.session_state.chat_history = []

    # Use session-stored chain
    qa_chain = st.session_state.qa_chain

    # Auto Insights
    st.subheader("📑 Case Summary Insights")
    questions = {
        "📝 Case Summary": "Summarize this legal case in 150 words.",
        "👥 Involved Parties": "Who are the parties involved in this case?",
        "⚖️ Verdict": "What was the final verdict or decision given by the court?",
        "📜 Legal Basis": "Which laws, legal provisions, or sections are cited in this case?"
    }

    for title, query in questions.items():
        with st.spinner(f"Analyzing: {title}..."):
            try:
                answer = qa_chain.run(query)
                st.markdown(f"**{title}**\n\n{answer}")
            except Exception as e:
                st.error(f"❌ Failed to analyze {title}: {e}")

    # 💬 Chat Assistant (Modern)
    st.subheader("💬 Ask Your Own Legal Questions")

    # Display chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for speaker, message in st.session_state.chat_history:
        with st.chat_message(name=speaker.lower()):
            st.markdown(message)

    # Chat input
    user_query = st.chat_input("Ask something about the case...")

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        try:
            response = qa_chain.run(user_query)
            with st.chat_message("assistant"):
                st.markdown(response)

            # Save to history
            st.session_state.chat_history.append(("User", user_query))
            st.session_state.chat_history.append(("Assistant", response))
        except Exception as e:
            st.error(f"❌ Failed to answer: {e}")

    # Clear chat
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
