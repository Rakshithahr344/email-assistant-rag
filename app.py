import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="AI Email Assistant", page_icon="✉️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1E88E5;'>✉️ Smart Email Writing Assistant (RAG Enabled)</h1>", unsafe_allow_html=True)

api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

if not api_key:
    st.info("Please enter your Google Gemini API Key in the sidebar to start.")
    st.stop()

@st.cache_resource
def init_rag(key):
    loader = TextLoader("data/email_examples.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
    return FAISS.from_documents(docs, embeddings)

try:
    vectorstore = init_rag(api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
except Exception as e:
    st.error(f"Error loading RAG: {e}")
    st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)

tab1, tab2 = st.tabs(["🚀 Generate Email", "🛠️ Refine & Edit"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        recipient = st.text_input("Recipient", placeholder="e.g., Hiring Manager")
        purpose = st.text_area("Purpose / Key Points", placeholder="e.g., Ask for budget approval")
        tone = st.selectbox("Tone", ["Professional", "Friendly", "Formal", "Persuasive"])
        btn = st.button("Generate Email")

    with col2:
        if btn and purpose and recipient:
            with st.spinner("Generating..."):
                docs = retriever.get_relevant_documents(f"Tone: {tone} Purpose: {purpose}")
                context = "\n".join([d.page_content for d in docs])
                template = """
                Context:
                {context}
                
                Write an email to {recipient} with tone '{tone}'.
                Purpose: {purpose}
                """
                prompt = PromptTemplate(template=template, input_variables=["context", "recipient", "tone", "purpose"])
                chain = LLMChain(llm=llm, prompt=prompt)
                res = chain.run(context=context, recipient=recipient, tone=tone, purpose=purpose)
                st.text_area("Result", value=res, height=250)

with tab2:
    text = st.text_area("Paste Email to Refine")
    c1, c2, c3 = st.columns(3)
    if c1.button("Rewrite") and text:
        st.write(llm.predict(f"Rewrite cleanly:\n\n{text}"))
    if c2.button("Shorten") and text:
        st.write(llm.predict(f"Shorten this email:\n\n{text}"))
    if c3.button("Fix Grammar") and text:
        st.write(llm.predict(f"Fix grammar:\n\n{text}"))
