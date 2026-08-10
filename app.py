import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="AI Email Assistant", page_icon="✉️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1E88E5;'>✉️ Smart Email Writing Assistant (RAG Enabled)</h1>", unsafe_allow_html=True)

# Fetch API key from Streamlit secrets or sidebar input
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Google Gemini API Key", type="password")

if not api_key:
    st.info("Please enter your Google Gemini API Key in the sidebar or Streamlit secrets to start.")
    st.stop()

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

# Robust model initialization with fallback
def generate_response(prompt_text, key):
    # Try gemini-1.5-flash-latest first, fall back to gemini-pro
    for model_name in ["gemini-1.5-flash-latest", "gemini-pro"]:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key,
                temperature=0.3
            )
            return llm.invoke([HumanMessage(content=prompt_text)]).content
        except Exception:
            continue
    raise Exception("All Gemini models failed. Please check your API Key in Google AI Studio.")

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

Write a complete email to {recipient} with a '{tone}' tone.
Purpose: {purpose}"""
                    
                    result_text = generate_response(prompt_text, api_key)
                    st.text_area("Result", value=result_text, height=300)
                except Exception as err:
                    st.error(f"Gemini API Call Failed: {err}")

with tab2:
    text = st.text_area("Paste Email to Refine")
    c1, c2, c3 = st.columns(3)
    if c1.button("Rewrite") and text:
        try:
            res = generate_response(f"Rewrite cleanly:\n\n{text}", api_key)
            st.write(res)
        except Exception as err:
            st.error(f"Error: {err}")
    if c2.button("Shorten") and text:
        try:
            res = generate_response(f"Shorten this email:\n\n{text}", api_key)
            st.write(res)
        except Exception as err:
            st.error(f"Error: {err}")
    if c3.button("Fix Grammar") and text:
        try:
            res = generate_response(f"Fix grammar:\n\n{text}", api_key)
            st.write(res)
        except Exception as err:
            st.error(f"Error: {err}")
