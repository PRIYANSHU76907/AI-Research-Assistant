import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📄"
)

st.title("📄 AI Research Assistant (RAG Prototype)")

st.write(
    "Upload a PDF and ask questions. "
    "The system retrieves the most relevant information from the document."
)

# -------------------------
# PDF UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# -------------------------
# PROCESS PDF
# -------------------------
if uploaded_file:

    try:

        # Read PDF
        pdf_reader = PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

        st.success("✅ PDF Processed Successfully!")

        # Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_text(text)

        st.info(f"📑 Total Chunks Created: {len(chunks)}")

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ChromaDB
        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings
        )

        st.success("✅ Embeddings Stored in ChromaDB")

        # Question
        question = st.text_input(
            "Ask a question about the document"
        )

        if question:

            with st.spinner("Searching document..."):

                docs = vector_db.similarity_search(
                    question,
                    k=3
                )

                answer = ""

                for doc in docs:
                    answer += doc.page_content + "\n\n"

                st.subheader("🤖 Retrieved Answer")

                st.write(answer[:1500])

    except Exception as e:

        st.error(f"Error: {str(e)}")