import streamlit as st
import os
from google import genai
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="AI Email Assistant", page_icon="✉️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1E88E5;'>✉️ Smart Email Writing Assistant (RAG Enabled)</h1>", unsafe_allow_html=True)

# Fetch API key from Streamlit secrets or sidebar input
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Google Gemini API Key", type="password")

if not api_key:
    st.info("Please enter your Google Gemini API Key in the sidebar or Streamlit secrets to start.")
    st.stop()

# Initialize Google GenAI client
client = genai.Client(api_key=api_key)

# Safe sample data creation
def get_or_create_sample_file():
    filepath = "email_examples.txt"
    if not os.path.exists(filepath):
        sample_data = """Subject: Follow-up on Proposal
Tone: Professional
Dear [Name],
I hope this email finds you well. I am following up on our previous discussion regarding the project proposal. Please let me know if you have any questions or require additional details.
Best regards,
[Sender]

Subject: Quick Question
Tone: Friendly
Hi [Name],
Hope you're having a great week! Just wanted to quickly check in regarding the update. Let me know when you have a moment to chat.
Cheers,
[Sender]
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sample_data)
    return filepath

sample_filepath = get_or_create_sample_file()

@st.cache_resource
def init_rag(filepath):
    loader = TextLoader(filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

try:
    vectorstore = init_rag(sample_filepath)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
except Exception as e:
    st.error(f"Error loading RAG: {e}")
    st.stop()

# Helper function using standard google-genai client
def generate_email(prompt_text):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt_text,
    )
    return response.text

tab1, tab2 = st.tabs(["🚀 Generate Email", "🛠️ Refine & Edit"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        recipient = st.text_input("Recipient", placeholder="e.g., HR Manager")
        purpose = st.text_area("Purpose / Key Points", placeholder="e.g., Requesting 2 days leave")
        tone = st.selectbox("Tone", ["Professional", "Friendly", "Formal", "Persuasive"])
        btn = st.button("Generate Email")

    with col2:
        if btn and purpose and recipient:
            with st.spinner("Generating..."):
                try:
                    docs = retriever.invoke(f"Tone: {tone} Purpose: {purpose}")
                    context = "\n".join([d.page_content for d in docs])
                    
                    prompt_text = f"""Context:
{context}

Write a clear, complete email to {recipient} with a '{tone}' tone.
Purpose: {purpose}"""
                    
                    result_text = generate_email(prompt_text)
                    st.text_area("Result", value=result_text, height=300)
                except Exception as err:
                    st.error(f"Gemini API Error: {err}")

with tab2:
    text = st.text_area("Paste Email to Refine")
    c1, c2, c3 = st.columns(3)
    if c1.button("Rewrite") and text:
        try:
            res = generate_email(f"Rewrite cleanly:\n\n{text}")
            st.write(res)
        except Exception as err:
            st.error(f"Error: {err}")
    if c2.button("Shorten") and text:
        try:
            res = generate_email(f"Shorten this email:\n\n{text}")
            st.write(res)
        except Exception as err:
            st.error(f"Error: {err}")
    if c3.button("Fix Grammar") and text:
        try:
            res = generate_email(f"Fix grammar:\n\n{text}")
            st.write(res)
        except Exception as err:
            st.error(f"Error: {err}")
