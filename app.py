import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📄"
)

st.title("📄 AI Research Assistant")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    st.success("✅ PDF Processed Successfully!")

    st.subheader("Extracted Text")

    st.write(text[:5000])

    question = st.text_input(
        "Ask a question about the document"
    )

    if question:

        st.subheader("Question")

        st.write(question)

        st.info(
            "RAG + Gemini integration can be added in the production version."
        )