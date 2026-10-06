import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from supabase import Client, create_client

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from google_sync import (
    decode_state,
    exchange_code,
    get_account_email,
    get_auth_url,
    make_flow,
    sync_calendar,
    sync_gmail,
)

load_dotenv()

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🏢", layout="wide")


# ---------- Config ----------
def get_secret(name, default=None):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return default


def get_google_config():
    """Google OAuth settings from env vars or the [google] block in secrets."""
    cid = os.getenv("GOOGLE_CLIENT_ID")
    secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect = os.getenv("GOOGLE_REDIRECT_URI")
    if cid and secret and redirect:
        return cid, secret, redirect
    try:
        g = st.secrets["google"]
        return g["client_id"], g["client_secret"], g["redirect_uri"]
    except Exception:
        return None


SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

if not (SUPABASE_URL and SUPABASE_KEY and OPENAI_API_KEY):
    st.error(
        "Missing credentials. Set SUPABASE_URL, SUPABASE_KEY and OPENAI_API_KEY "
        "in `.env` or Streamlit secrets."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


@st.cache_resource
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


supabase = get_supabase()


# ---------- Vector store ----------
class SupabaseClientVectorStore:
    """Multi-tenant pgvector store: every row is tagged with a client_id."""

    TABLE = "client_knowledge_base"
    BATCH = 100

    def __init__(self, client_name: str, embeddings):
        self.client_id = client_name.strip().lower().replace(" ", "_")
        self.embeddings = embeddings
        self.db = supabase

    def add_texts(self, texts, metadatas=None) -> int:
        pairs = [
            (t, (metadatas[i] if metadatas else {}))
            for i, t in enumerate(texts)
            if t and t.strip()
        ]
        if not pairs:
            return 0
        vectors = self.embeddings.embed_documents([t for t, _ in pairs])
        rows = [
            {
                "client_id": self.client_id,
                "content": text,
                "metadata": meta,
                "embedding": vec,
            }
            for (text, meta), vec in zip(pairs, vectors)
        ]
        for i in range(0, len(rows), self.BATCH):  # batched inserts
            self.db.table(self.TABLE).insert(rows[i : i + self.BATCH]).execute()
        return len(rows)

    def add_documents(self, docs) -> int:
        return self.add_texts(
            [d.page_content for d in docs], [d.metadata for d in docs]
        )

    def count(self) -> int:
        try:
            res = (
                self.db.table(self.TABLE)
                .select("id", count="exact")
                .eq("client_id", self.client_id)
                .execute()
            )
            return res.count or 0
        except Exception:
            return 0

    def search(self, query: str, k: int = 3):
        try:
            emb = self.embeddings.embed_query(query)
            res = self.db.rpc(
                "match_client_documents",
                {
                    "query_embedding": emb,
                    "match_threshold": 0.2,
                    "match_count": k,
                    "p_client_id": self.client_id,
                },
            ).execute()
            return res.data or []
        except Exception as e:
            st.error(f"Vector search error: {e}")
            return []


# ---------- Chain ----------
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional, helpful assistant for {client}.\n"
            "Use the retrieved context below to answer questions accurately.\n"
            "If the answer is not in the context, politely say it is outside the "
            "knowledge base.\n"
            "Keep your answers concise and professional.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def answer(client: str, question: str, chunks: list) -> str:
    context = (
        "\n\n".join(c["content"] for c in chunks)
        if chunks
        else "No matching documents found in client knowledge base."
    )
    chain = PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"client": client, "question": question, "context": context})


# ---------- Google OAuth return handling (runs before widgets) ----------
google_cfg = get_google_config()

if "client_name" not in st.session_state:
    st.session_state["client_name"] = "Demo Company"

params = st.query_params
if google_cfg and "code" in params and "google_creds" not in st.session_state:
    try:
        flow = make_flow(*google_cfg)
        st.session_state["google_creds"] = exchange_code(flow, params["code"])
        st.session_state["google_email"] = get_account_email(
            st.session_state["google_creds"]
        )
        restored = decode_state(params.get("state", ""))
        if restored:
            st.session_state["client_name"] = restored  # keep the workspace
    except Exception as e:
        st.session_state["google_error"] = str(e)
    st.query_params.clear()
    st.rerun()


# ---------- UI ----------
st.title("🏢 Enterprise Client RAG Portal")
st.markdown(
    "Create or select a client workspace, upload documents, sync Google data, "
    "and chat against that client's knowledge base."
)

with st.sidebar:
    st.header("⚙️ Workspace Setup")
    client_name = st.text_input("Client / Company Name", key="client_name", max_chars=50)
    if not client_name.strip():
        st.warning("Please enter a client name to proceed.")
        st.stop()

    vector_store = SupabaseClientVectorStore(client_name, get_embeddings())

    st.divider()
    st.header("📂 Ingest Knowledge Base")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    uploaded = st.file_uploader("Upload Client PDF Document", type=["pdf"])
    if uploaded is not None and st.button("Process & Save PDF"):
        with st.spinner("Parsing and vectorizing PDF..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            try:
                pages = PyPDFLoader(tmp_path).load()
            finally:
                os.remove(tmp_path)
            for p in pages:
                p.metadata = {
                    "source": uploaded.name,
                    "page": p.metadata.get("page"),
                }
            docs = splitter.split_documents(pages)
            added = vector_store.add_documents(docs)
        st.success(f"Added {added} chunks to {client_name}.")
        st.rerun()

    st.markdown("---")
    raw_text = st.text_area(
        "Or paste custom FAQ / policy text",
        placeholder="Type or paste company data here...",
    )
    if st.button("Save Text"):
        if raw_text.strip():
            with st.spinner("Vectorizing text..."):
                docs = splitter.create_documents(
                    [raw_text], metadatas=[{"source": "pasted_text"}]
                )
                added = vector_store.add_documents(docs)
            st.success(f"Saved {added} chunks to {client_name}.")
            st.rerun()
        else:
            st.error("Please enter some text before saving.")

    st.divider()

    with st.expander("🌐 Google Cloud Sync"):
        if not google_cfg:
            st.info(
                "Google sync is not configured. Add a `[google]` block with "
                "`client_id`, `client_secret` and `redirect_uri` to your secrets."
            )
        else:
            if st.session_state.get("google_error"):
                st.error(f"Google sign-in failed: {st.session_state.pop('google_error')}")

            creds = st.session_state.get("google_creds")
            if not creds:
                flow = make_flow(*google_cfg)
                st.link_button(
                    "Sign in with Google", get_auth_url(flow, client_name)
                )
                st.caption(
                    "Sign-in opens in a new tab and continues there. "
                    "You can pick any Google account."
                )
            else:
                st.success(f"Signed in as {st.session_state.get('google_email', 'Google user')}")
                email_limit = st.slider("Max emails to sync", 1, 20, 5)
                cal_limit = st.slider("Max calendar events to sync", 1, 20, 10)

                if st.button("Sync Gmail & Calendar"):
                    with st.spinner("Syncing..."):
                        try:
                            n_mail = sync_gmail(vector_store, creds, email_limit)
                            n_cal = sync_calendar(vector_store, creds, cal_limit)
                            st.success(f"Synced {n_mail} emails and {n_cal} events.")
                        except Exception as e:
                            st.error(f"Google sync failed: {e}")

                if st.button("Sign out of Google"):
                    st.session_state.pop("google_creds", None)
                    st.session_state.pop("google_email", None)
                    st.rerun()

    st.divider()
    st.metric(f"Knowledge chunks ({client_name})", vector_store.count())


# ---------- Chat ----------
st.subheader(f"💬 Live Support Chat: {client_name}")

history_key = f"messages_{vector_store.client_id}"
st.session_state.setdefault(history_key, [])

for msg in st.session_state[history_key]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input(f"Ask a question about {client_name}..."):
    st.session_state[history_key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching records and generating response..."):
            chunks = vector_store.search(question, k=3)  # retrieval runs once
            reply = answer(client_name, question, chunks)
            st.markdown(reply)

            with st.expander("🔍 Retrieved source chunks (debug)"):
                if chunks:
                    for i, c in enumerate(chunks, 1):
                        meta = c.get("metadata") or {}
                        label = meta.get("source", "unknown")
                        if meta.get("page") is not None:
                            label += f" · page {meta['page'] + 1}"
                        st.markdown(f"**Chunk {i}** — {label}")
                        st.text(c["content"])
                else:
                    st.info("No relevant chunks retrieved for this query.")

    st.session_state[history_key].append({"role": "assistant", "content": reply})
