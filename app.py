
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# PDF directory
pdf_dir = "pdfs"
chroma_dir = "chromadb_store"

# Check if ChromaDB already exists
if not os.path.exists(chroma_dir):
    st.info("Building knowledge base from PDFs...")

    # Collect all readable chunks
    all_chunks = []
    failed = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(pdf_dir, filename))
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
            except Exception as e:
                failed.append(filename)

    # Embed and store
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(all_chunks, embedding, persist_directory=chroma_dir)
    db.persist()
    st.success(f"Knowledge base built. {len(all_chunks)} chunks added.")
    if failed:
        st.warning(f"Skipped {len(failed)} unreadable PDFs.")
else:
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma(persist_directory=chroma_dir, embedding_function=embedding)

# Retrieval setup
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key),
    retriever=retriever,
    return_source_documents=True
)

# UI
st.title("📚 AskMyPharm")
st.caption("Query your neonatal and pediatric research papers.")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Searching your documents..."):
        result = qa(query)
        st.markdown("### 📄 Answer:")
        st.write(result["result"])

        st.markdown("### 🔍 Sources:")
        for doc in result["source_documents"]:
            st.markdown(f"- `{doc.metadata.get('source', 'Unknown')}`")
