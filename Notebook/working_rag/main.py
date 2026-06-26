import os
import shutil
import json
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Modern Google GenAI SDK
from google import genai

# Initialize Client (automatically detects GOOGLE_API_KEY from environment)
client = genai.Client()

from sentence_transformers import SentenceTransformer
import faiss

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------
# Streamlit UI Configuration
# -------------------------------
st.set_page_config(page_title="Oaklane Capital Research Analyst AI", layout="wide")
st.title("📊 Research Analyst AI")
# -------------------------------
# UI Guide & Overview
# -------------------------------
st.markdown("""
###  How This Helps You
As a Research Analyst, keeping track of moving market data across multiple long-form articles is time-consuming. This tool automates that workflow by **instantly reading and indexing financial news, press releases, or market reports**, letting you extract verified answers in seconds without manual reading.

---

###  How to Use the Tool

1. **Provide the Sources (Left Sidebar):**
   * **What URLs to paste:** Input links to financial articles, company earnings reports, market analyses, or news columns (e.g., Bloomberg, Reuters, CNBC, or Yahoo Finance links).
   * Click **"Process URLs"** to split the text, build a secure local knowledge base, and save your references.

2. **Ask Targeted Questions (Main Screen):**
   * **What questions to ask:** Ask specific, data-driven financial questions based on your provided URLs. 
   * *Examples:* * *"What was the company's net profit margin for Q3?"*
     * *"What reasons did the CEO give for the drop in hardware revenue?"*
     * *"List the three main risk factors mentioned regarding the new supply chain."*

3. **Get Verified Answers & Sources:**
   * The AI will generate a strict, context-backed response. 
   * Instead of dumping massive blocks of text on your screen, it lists the **exact URL sources** underneath the answer so you can quickly double-check facts and build your final reports.
""")
st.markdown("---")
st.sidebar.header("News URLs")
urls = []
for i in range(3):
    urls.append(st.sidebar.text_input(f"URL {i+1}"))

process = st.sidebar.button("Process URLs")

# Load embedding model globally
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Action: Process URLs
# -------------------------------
if process:
    # Filter out empty inputs
    active_urls = [url for url in urls if url.strip()]
    
    if not active_urls:
        st.sidebar.error("Please provide at least one URL.")
    else:
        st.info("Loading articles and extracting content...")
        loader = UnstructuredURLLoader(urls=active_urls)
        docs = loader.load()

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # Extract text strings and track source URLs mapped by LangChain metadata
        texts = [chunk.page_content for chunk in chunks]
        sources = [chunk.metadata.get("source", "Unknown Source") for chunk in chunks]

        # Generate embeddings
        st.info("Generating Vector Embeddings...")
        embeddings = embedding_model.encode(texts)
        dimension = embeddings.shape[1]

        # Build FAISS Index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Recreate clean local database directory
        if os.path.exists("vector_db"):
            shutil.rmtree("vector_db")
        os.mkdir("vector_db")

        # Save Vector Index
        faiss.write_index(index, "vector_db/index.faiss")

        # Map text to sources inside JSON knowledge base
        knowledge_base = [{"text": t, "source": s} for t, s in zip(texts, sources)]
        with open("vector_db/knowledge_base.json", "w", encoding="utf8") as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

        st.success("Knowledge Base Created Successfully!")

# -------------------------------
# Action: Ask Question
# -------------------------------
question = st.text_input("Ask a Question")

if question:
    if not os.path.exists("vector_db/index.faiss") or not os.path.exists("vector_db/knowledge_base.json"):
        st.error("Please process your URLs first to establish the database context.")
    else:
        # Load local FAISS index and JSON mappings
        index = faiss.read_index("vector_db/index.faiss")
        with open("vector_db/knowledge_base.json", "r", encoding="utf8") as f:
            knowledge_base = json.load(f)

        # Search nearest vectors
        q_embedding = embedding_model.encode([question])
        D, I = index.search(q_embedding, 5)

        context = ""
        matched_sources = set()  # Set prevents duplicate sources listing 

        for idx in I[0]:
            # Guard against edge-cases where index might map out of range
            if idx < len(knowledge_base):
                chunk_data = knowledge_base[idx]
                context += chunk_data["text"] + "\n"
                matched_sources.add(chunk_data["source"])

        # Construct LLM prompt boundary
        prompt = f"""
You are an expert financial research analyst.

Answer ONLY using the context below.

If the answer is not available, say
"I could not find this information."

Context:
{context}

Question:
{question}
"""

        # Generate model inference using unified Interactions API interface
        with st.spinner("Analyzing data via Gemini..."):
            interaction = client.interactions.create(
                model="gemini-2.5-flash",
                input=prompt
            )

        # Output UI Elements
        st.subheader("Answer")
        st.write(interaction.output_text)

        st.subheader("Sources Used")
        for source in matched_sources:
            if source.startswith("http"):
                st.markdown(f"- [{source}]({source})")
            else:
                st.markdown(f"- {source}")