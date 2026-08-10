# ✉️ AI Email Writing Assistant (RAG-Enabled)

A smart, context-aware Email Writing Assistant built using **Retrieval-Augmented Generation (RAG)**, **LangChain**, **FAISS**, **Google Gemini API**, and **Streamlit**. 

This application retrieves pre-conditioned email templates and structural context from a custom knowledge base to generate highly relevant emails based on user-defined parameters, alongside dedicated tools for refining existing drafts.

---

## 🌟 Key Features

* **RAG-Powered Generation:** Leverages FAISS vector search to retrieve structural examples based on recipient, context, and selected tone.
* **Multiple Writing Tones:** Tailor outputs for *Professional*, *Friendly*, *Formal*, or *Persuasive* settings.
* **Email Editing & Refinement Tools:**
  * 🔄 **Rewrite:** Improve flow and clarity.
  * ✂️ **Shorten:** Condense long drafts into concise messages.
  * ✏️ **Fix Grammar:** Correct typos, syntax, and grammatical errors.
  * 🌐 **Translate:** Convert emails into target languages.
* **Modern UI:** Clean, dual-tab layout built with Streamlit.

---

## 🛠️ Tech Stack

* **Frontend/UI:** [Streamlit](https://streamlit.io/)
* **LLM & Embeddings:** [Google Gemini API](https://ai.google.dev/) (`gemini-1.5-flash` & `embedding-001`)
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)

---

## 📁 Project Structure

```text
email-assistant-rag/
├── data/
│   └── email_examples.txt   # Knowledge base for RAG vector index
├── app.py                   # Main Streamlit application file
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
