import os
import streamlit as st
import openai
from supabase import create_client, Client
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env
load_dotenv()

# Page Config
st.set_page_config(
    page_title="Enterprise RAG Assistant", page_icon="🏢", layout="wide"
)

# --- LOAD SECRETS FROM .ENV ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    st.error("Missing credentials in your `.env` file. Please ensure SUPABASE_URL, SUPABASE_KEY, and OPENAI_API_KEY are all set.")
    st.stop()

# Initialize Supabase and OpenAI clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)


class SupabaseClientVectorStore:
    """An isolated, permanent vector store backed by Supabase pgvector table with multi-tenant filtering."""

    def __init__(self, client_name, embeddings_model):
        self.client_id = client_name.strip().lower().replace(" ", "_")
        self.embeddings_model = embeddings_model
        self.client = supabase

    def add_documents(self, docs):
        texts = [doc.page_content for doc in docs if doc.page_content.strip()]
        if not texts:
            return
        
        # Embed documents using OpenAIEmbeddings
        embeddings = self.embeddings_model.embed_documents(texts)
        
        # Insert each chunk into Supabase
        for text, emb in zip(texts, embeddings):
            data = {
                "client_id": self.client_id,
                "content": text,
                "metadata": {"source": "langchain_upload"},
                "embedding": emb
            }
            supabase.table("client_knowledge_base").insert(data).execute()

    def get_document_count(self):
        try:
            response = supabase.table("client_knowledge_base") \
                .select("id", count="exact") \
                .eq("client_id", self.client_id) \
                .execute()
            return response.count if response.count is not None else 0
        except Exception:
            return 0

    def similarity_search(self, query, k=3):
        # 1. Embed user query
        query_emb = self.embeddings_model.embed_query(query)
        
        try:
            # 2. Call Supabase RPC function for fast vector search
            rpc_response = supabase.rpc(
                "match_client_documents",
                {
                    "query_embedding": query_emb,
                    "match_threshold": 0.2,
                    "match_count": k,
                    "p_client_id": self.client_id
                }
            ).execute()
            
            matches = rpc_response.data
            return [match["content"] for match in matches] if matches else []
        except Exception as e:
            st.error(f"Vector search error: {e}")
            return []


# --- UI Layout ---
st.title("🏢 Enterprise Client RAG Portal")
st.markdown(
    "Demonstration system: Create or select a client workspace, upload documents, sync personal Google data, and query their dedicated Supabase knowledge base."
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

    # Initialize client-specific vector store wrapper
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = SupabaseClientVectorStore(client_name, embeddings)

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

        if st.button("Process & Save PDF to Supabase"):
            with st.spinner("Parsing and vectorizing PDF into cloud storage..."):
                loader = PyPDFLoader(temp_pdf_path)
                pages = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50
                )
                docs = text_splitter.split_documents(pages)
                vector_store.add_documents(docs)
                os.remove(temp_pdf_path)
            st.success(f"Added {len(docs)} chunks to {client_name}'s cloud memory!")
            st.rerun()

    st.markdown("---")

    # Option 2: Paste Raw Text / Custom FAQ
    raw_text_input = st.text_area(
        "Or Paste Custom FAQ / Policy Text",
        placeholder="Type or paste company data here...",
    )
    if st.button("Save Raw Text to Cloud Memory"):
        if raw_text_input.strip():
            with st.spinner("Vectorizing text..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50
                )
                docs = text_splitter.create_documents([raw_text_input])
                vector_store.add_documents(docs)
            st.success(
                f"Successfully saved text chunk to {client_name}'s cloud memory!"
            )
            st.rerun()
        else:
            st.error("Please enter some text before saving.")

    st.divider()

    # Option 3: Google Workspace Cloud Sync
    with st.expander("🌐 Google Cloud Sync"):
        st.markdown("Pull live data from your personal Gmail & Calendar into **" + client_name + "**.")
        
        email_limit = st.slider("Max Emails to Sync", 1, 20, 5)
        calendar_limit = st.slider("Max Calendar Events to Sync", 1, 20, 10)

        if st.button("Sync Gmail & Calendar"):
            with st.spinner("Connecting to Google APIs... (Check your browser for OAuth sign-in if prompted)"):
                try:
                    from google_sync import sync_live_gmail_to_supabase, sync_live_calendar_to_supabase
                    
                    # Run syncs for the current active client/workspace
                    emails_count = sync_live_gmail_to_supabase(vector_store, openai_client, max_results=email_limit)
                    events_count = sync_live_calendar_to_supabase(vector_store, openai_client, max_results=calendar_limit)
                    
                    st.success(f"Synced {emails_count} emails and {events_count} events successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Google sync failed: {e}")

    st.divider()
    st.metric(
        label=f"Active Knowledge Chunks ({client_name})",
        value=vector_store.get_document_count(),
    )


# --- Build Agent for Active Client ---
@st.cache_resource
def get_support_agent(client_key):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    store = SupabaseClientVectorStore(client_key, OpenAIEmbeddings(model="text-embedding-3-small"))

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
        f"You are a professional, helpful assistant for {client_key}.\n"
        "Use the retrieved context below to answer questions accurately.\n"
        "If you do not know the answer based on the context, politely inform the user that it is outside the knowledge base.\n"
        "Keep your answers concise and professional.\n\nContext: {context}"
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
            # 1. Fetch retrieved chunks explicitly so we can inspect them
            retrieved_chunks = vector_store.similarity_search(prompt, k=3)
            
            # 2. Generate response using the agent
            response = agent.invoke(prompt)
            
            # 3. Display the answer
            st.markdown(response)
            
            # 4. Debug Expander to see what source chunks were used!
            with st.expander("🔍 View Retrieved Source Chunks (Debug)"):
                if retrieved_chunks:
                    for i, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text(chunk)
                else:
                    st.info("No relevant chunks retrieved for this query.")

    st.session_state[session_key].append(
        {"role": "assistant", "content": response}
    )


SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")