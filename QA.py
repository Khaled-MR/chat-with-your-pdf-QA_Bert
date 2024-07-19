import streamlit as st
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Function to extract text from a PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Load the trained model and tokenizer from Hugging Face
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "khaledsayed1/Question_answering_bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, use_auth_token="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")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

# Streamlit app
st.set_page_config(page_title="PDF Question Answering", page_icon="📄", layout="wide")
st.title("📄 PDF Question Answering")
st.write("Upload a PDF file and ask questions about its content using a pretrained BERT model.")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        # Extract text from the uploaded PDF file
        context = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text", context, height=200)

    # Load the trained model and tokenizer from Hugging Face
    qa_pipeline = load_model()

    # Input questions
    question = st.text_input("Enter your question")

    if st.button("Get Answer"):
        if question:
            with st.spinner("Finding answer..."):
                # Perform inference
                result = qa_pipeline(question=question, context=context)
                st.write(f"**Question:** {question}")
                st.write(f"**Answer:** {result['answer']}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a PDF file to get started.")

# Footer
st.markdown("""
---
Developed by [Your Name](https://www.yourwebsite.com)
""")
