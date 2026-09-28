import json
import os
import sqlite3
import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page Config
st.set_page_config(
    page_title="Enterprise RAG Assistant", page_icon="🏢", layout="wide"
)

# Ensure storage directory exists
DB_DIR = "./client_databases"
os.makedirs(DB_DIR, exist_ok=True)


def cosine_similarity(a, b):
  """Computes cosine similarity between two vectors."""
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class ClientVectorStore:
  """An isolated, permanent vector store backed by a client-specific SQLite file."""

  def __init__(self, client_name, embeddings_model):
    self.client_name = client_name.strip().lower().replace(" ", "_")
    self.db_path = os.path.join(DB_DIR, f"{self.client_name}_rag.db")
    self.embeddings_model = embeddings_model
    self._init_db()

  def _init_db(self):
    self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
    self.cursor = self.conn.cursor()
    self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
        """)
    self.conn.commit()

  def add_documents(self, docs):
    texts = [doc.page_content for doc in docs if doc.page_content.strip()]
    if not texts:
      return
    embeddings = self.embeddings_model.embed_documents(texts)
    for text, emb in zip(texts, embeddings):
      emb_json = json.dumps(emb)
      self.cursor.execute(
          "INSERT INTO knowledge_base (text, embedding) VALUES (?, ?)",
          (text, emb_json),
      )
    self.conn.commit()

  def get_document_count(self):
    self.cursor.execute("SELECT COUNT(*) FROM knowledge_base")
    return self.cursor.fetchone()[0]

  def similarity_search(self, query, k=3):
    query_emb = self.embeddings_model.embed_query(query)
    self.cursor.execute("SELECT text, embedding FROM knowledge_base")
    rows = self.cursor.fetchall()

    if not rows:
      return []

    scored_docs = []
    for text, emb_json in rows:
      db_emb = json.loads(emb_json)
      score = cosine_similarity(query_emb, db_emb)
      scored_docs.append((score, text))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored_docs[:k]]


# --- UI Layout ---
st.title("🏢 Enterprise Client RAG Portal")
st.markdown(
    "Demonstration system: Create or select a client workspace, upload"
    " documents, and query their dedicated knowledge base."
)

# Sidebar for Client Workspace Setup
with st.sidebar:
  st.header("⚙️ Workspace Setup")
  client_name = st.text_input(
      "Client / Company Name", value="Demo Company", max_chars=50
  )

  if not client_name:
    st.warning("Please enter a client name to proceed.")
    st.stop()

  # Initialize client-specific vector store
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  vector_store = ClientVectorStore(client_name, embeddings)

  st.divider()
  st.header("📂 Ingest Knowledge Base")

  # Option 1: Upload PDF
  uploaded_file = st.file_uploader(
      "Upload Client PDF Document", type=["pdf"]
  )
  if uploaded_file is not None:
    temp_pdf_path = os.path.join(".", uploaded_file.name)
    with open(temp_pdf_path, "wb") as f:
      f.write(uploaded_file.getbuffer())

    if st.button("Process & Save PDF to Permanent Storage"):
      with st.spinner("Parsing and vectorizing PDF..."):
        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        docs = text_splitter.split_documents(pages)
        vector_store.add_documents(docs)
        os.remove(temp_pdf_path)
      st.success(f"Added {len(docs)} chunks to {client_name}'s memory!")
      st.rerun()

  st.markdown("---")

  # Option 2: Paste Raw Text / Custom FAQ
  raw_text_input = st.text_area(
      "Or Paste Custom FAQ / Policy Text",
      placeholder="Type or paste company data here...",
  )
  if st.button("Save Raw Text to Memory"):
    if raw_text_input.strip():
      with st.spinner("Vectorizing text..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        docs = text_splitter.create_documents([raw_text_input])
        vector_store.add_documents(docs)
      st.success(
          f"Successfully saved text chunk to {client_name}'s permanent memory!"
      )
      st.rerun()
    else:
      st.error("Please enter some text before saving.")

  st.divider()
  st.metric(
      label=f"Active Knowledge Chunks ({client_name})",
      value=vector_store.get_document_count(),
  )


# --- Build Agent for Active Client ---
@st.cache_resource
def get_support_agent(client_key):
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
  # Re-instantiate store for the runnable
  store = ClientVectorStore(client_key, OpenAIEmbeddings(model="text-embedding-3-small"))

  retrieve_runnable = RunnableLambda(
      lambda query: store.similarity_search(query, k=3)
  )
  format_runnable = RunnableLambda(
      lambda docs: (
          "\n\n".join(docs)
          if docs
          else "No matching documents found in client knowledge base."
      )
  )

  system_prompt = (
      f"You are a professional, helpful customer support agent for"
      f" {client_name}.\nUse the retrieved context below to answer customer"
      " questions accurately.\nIf you do not know the answer based on the"
      " context, politely inform the user that it is outside the company's"
      " knowledge base.\nKeep your answers concise and professional.\n\nContext:"
      " {context}"
  )

  prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("human", "{question}"),
  ])

  chain = (
      {
          "context": retrieve_runnable | format_runnable,
          "question": RunnablePassthrough(),
      }
      | prompt
      | llm
      | StrOutputParser()
  )
  return chain


agent = get_support_agent(client_name)

# --- Main Chat Interface ---
st.subheader(f"💬 Live Support Chat: {client_name}")

# Maintain separate session state history per client
session_key = f"messages_{client_name}"
if session_key not in st.session_state:
  st.session_state[session_key] = []

# Display chat history
for message in st.session_state[session_key]:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# User prompt input
if prompt := st.chat_input(f"Ask a question about {client_name}..."):
  st.session_state[session_key].append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    with st.spinner("Searching records and generating response..."):
      response = agent.invoke(prompt)
      st.markdown(response)

  st.session_state[session_key].append(
      {"role": "assistant", "content": response}
  )