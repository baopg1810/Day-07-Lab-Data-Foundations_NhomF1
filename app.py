"""
Streamlit Demo UI — Day 7 Lab: Embedding & Vector Store
Run: streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Demo — Day 7 Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load .env (provider selection) ───────────────────────────────────────────
load_dotenv(override=False)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* Main background */
.stApp { background: #0f172a; color: #e2e8f0; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #38bdf8; }
.metric-card .label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }

/* Chunk result card */
.chunk-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 0.875rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    line-height: 1.6;
}
.chunk-card .score-badge {
    display: inline-block;
    background: #0ea5e9;
    color: #fff;
    border-radius: 20px;
    padding: 1px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.chunk-card .source-tag {
    display: inline-block;
    background: #334155;
    color: #94a3b8;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.72rem;
    margin-left: 6px;
}
.chunk-card .content { color: #cbd5e1; }

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, #0c4a6e22, #0f172a);
    border: 1px solid #0ea5e9;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    font-size: 0.92rem;
    line-height: 1.75;
    color: #e2e8f0;
}

/* Section header */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin: 1.5rem 0 0.5rem 0;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.3rem;
}

/* Doc pill */
.doc-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #94a3b8;
    margin: 2px;
}

/* Strategy compare table */
.compare-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
.compare-cell {
    flex: 1;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.75rem;
    text-align: center;
}
.compare-cell .strat-name { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.compare-cell .strat-val { font-size: 1.4rem; font-weight: 700; color: #f1f5f9; }

/* Inputs override */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

div[data-testid="stFileUploader"] {
    background: #1e293b;
    border: 1px dashed #334155;
    border-radius: 10px;
    padding: 0.5rem;
}

/* Tab styling */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
ALLOWED_EXT = {".md", ".txt"}


def _get_embedder(provider: str):
    from src.embeddings import (
        LOCAL_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL,
        LocalEmbedder, OpenAIEmbedder, _mock_embed,
    )
    if provider == "local":
        try:
            emb = LocalEmbedder(model_name=LOCAL_EMBEDDING_MODEL)
            return emb, emb._backend_name
        except Exception as e:
            st.warning(f"Local embedder failed ({e}), falling back to mock.")
            return _mock_embed, "mock (fallback)"
    elif provider == "openai":
        try:
            emb = OpenAIEmbedder(model_name=OPENAI_EMBEDDING_MODEL)
            return emb, emb._backend_name
        except Exception as e:
            st.warning(f"OpenAI embedder failed ({e}), falling back to mock.")
            return _mock_embed, "mock (fallback)"
    else:
        return _mock_embed, "mock (deterministic)"


def _load_data_dir_docs(selected_names: list[str]) -> list:
    from src.models import Document
    docs = []
    for name in selected_names:
        path = DATA_DIR / name
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        docs.append(Document(
            id=path.stem,
            content=content,
            metadata={"source": name, "extension": path.suffix.lower(), "doc_id": path.stem},
        ))
    return docs


def _load_uploaded_docs(uploaded_files) -> list:
    from src.models import Document
    docs = []
    for f in uploaded_files:
        ext = Path(f.name).suffix.lower()
        if ext not in ALLOWED_EXT:
            st.sidebar.warning(f"Bỏ qua {f.name} (chỉ hỗ trợ .txt/.md)")
            continue
        content = f.read().decode("utf-8")
        doc_id = Path(f.name).stem
        docs.append(Document(
            id=doc_id,
            content=content,
            metadata={"source": f.name, "extension": ext, "doc_id": doc_id},
        ))
    return docs


def _chunk_docs(docs, strategy: str, chunk_size: int, max_sentences: int):
    from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker
    from src.models import Document

    if strategy == "FixedSize":
        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_size // 10)
    elif strategy == "Sentence":
        chunker = SentenceChunker(max_sentences_per_chunk=max_sentences)
    else:
        chunker = RecursiveChunker(chunk_size=chunk_size)

    chunked: list[Document] = []
    for doc in docs:
        pieces = chunker.chunk(doc.content)
        for idx, piece in enumerate(pieces):
            chunked.append(Document(
                id=f"{doc.id}_chunk{idx}",
                content=piece,
                metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": idx},
            ))
    return chunked


def _build_store(chunks, embedder):
    from src.store import EmbeddingStore
    store = EmbeddingStore(collection_name="demo_store", embedding_fn=embedder)
    store.add_documents(chunks)
    return store


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [
    ("store", None),
    ("embedder", None),
    ("backend_name", ""),
    ("docs_loaded", []),
    ("chunks_count", 0),
    ("indexed", False),
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 RAG Demo")
    st.markdown("<div class='section-header'>Embedding Backend</div>", unsafe_allow_html=True)
    provider_choice = st.radio(
        "Provider",
        ["mock", "local", "openai"],
        index=["mock", "local", "openai"].index(os.getenv("EMBEDDING_PROVIDER", "mock").lower()),
        label_visibility="collapsed",
    )

    st.markdown("<div class='section-header'>Chunking Strategy</div>", unsafe_allow_html=True)
    strategy = st.selectbox(
        "Strategy", ["FixedSize", "Sentence", "Recursive"], label_visibility="collapsed"
    )
    if strategy in ("FixedSize", "Recursive"):
        chunk_size = st.slider("Chunk size (chars)", 100, 1000, 300, 50)
    else:
        chunk_size = 300
    if strategy == "Sentence":
        max_sentences = st.slider("Max sentences / chunk", 1, 10, 3)
    else:
        max_sentences = 3

    st.markdown("<div class='section-header'>Documents</div>", unsafe_allow_html=True)

    # List files in data/
    data_files = sorted(
        f.name for f in DATA_DIR.glob("*")
        if f.suffix.lower() in ALLOWED_EXT
    ) if DATA_DIR.exists() else []

    selected_data = st.multiselect(
        "Chọn file từ data/",
        data_files,
        default=data_files,
        label_visibility="collapsed",
    )

    uploaded = st.file_uploader(
        "Hoặc upload file (.txt / .md)",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    st.markdown("")
    index_btn = st.button("⚡ Index Documents", use_container_width=True)

    if st.session_state.indexed:
        st.markdown(f"""
        <div style='margin-top:1rem;background:#0f2a1a;border:1px solid #166534;border-radius:8px;padding:0.6rem 1rem;font-size:0.8rem;'>
        ✅ <b style='color:#4ade80;'>Indexed</b><br>
        <span style='color:#64748b;'>{st.session_state.chunks_count} chunks · {st.session_state.backend_name}</span>
        </div>
        """, unsafe_allow_html=True)

    if index_btn:
        with st.spinner("Loading & indexing documents…"):
            embedder, backend_name = _get_embedder(provider_choice)
            all_docs = _load_data_dir_docs(selected_data) + _load_uploaded_docs(uploaded or [])

            if not all_docs:
                st.error("Không có document nào được load!")
            else:
                chunks = _chunk_docs(all_docs, strategy, chunk_size, max_sentences)
                store = _build_store(chunks, embedder)

                st.session_state.store = store
                st.session_state.embedder = embedder
                st.session_state.backend_name = backend_name
                st.session_state.docs_loaded = all_docs
                st.session_state.chunks_count = store.get_collection_size()
                st.session_state.indexed = True
                st.session_state.chat_history = []
                st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#475569;text-align:center;'>Day 7 · Data Foundations<br>Embedding & Vector Store</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🔍 RAG Knowledge Base Demo")

if not st.session_state.indexed:
    # Welcome screen
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0ea5e922,#6366f122);border:1px solid #334155;
                border-radius:16px;padding:2rem 2.5rem;margin-top:1rem;'>
        <h3 style='color:#38bdf8;margin:0 0 0.75rem 0;'>Bắt đầu bằng cách Index Documents</h3>
        <p style='color:#94a3b8;margin:0;font-size:0.9rem;'>
            1. Chọn <b>Embedding Backend</b> (mock / local / openai) ở sidebar<br>
            2. Chọn <b>Chunking Strategy</b> và điều chỉnh tham số<br>
            3. Chọn files từ <code>data/</code> hoặc upload file mới<br>
            4. Nhấn <b>⚡ Index Documents</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show data files preview
    if data_files:
        st.markdown("<div class='section-header' style='margin-top:2rem;'>Files có sẵn trong data/</div>", unsafe_allow_html=True)
        cols = st.columns(4)
        for i, name in enumerate(data_files):
            size = (DATA_DIR / name).stat().st_size
            cols[i % 4].markdown(f"""
            <div class='metric-card' style='margin-bottom:0.5rem;'>
                <div style='font-size:0.85rem;color:#e2e8f0;font-weight:500;'>{name}</div>
                <div style='font-size:0.75rem;color:#475569;margin-top:0.25rem;'>{size/1024:.1f} KB</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_search, tab_chat, tab_compare, tab_docs = st.tabs([
        "🔎 Search", "💬 Agent Q&A", "📊 Strategy Compare", "📄 Documents"
    ])

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class='metric-card'>
            <div class='value'>{len(st.session_state.docs_loaded)}</div>
            <div class='label'>Documents</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='metric-card'>
            <div class='value'>{st.session_state.chunks_count}</div>
            <div class='label'>Chunks</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='metric-card'>
            <div class='value'>{strategy}</div>
            <div class='label'>Strategy</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class='metric-card'>
            <div class='value' style='font-size:1rem;padding-top:0.4rem;'>{st.session_state.backend_name[:18]}</div>
            <div class='label'>Embedder</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: SEARCH
    # ─────────────────────────────────────────────────────────────────────────
    with tab_search:
        st.markdown("### Vector Search")
        col_q, col_k = st.columns([5, 1])
        with col_q:
            query = st.text_input("Nhập câu hỏi hoặc từ khóa…", key="search_query",
                                  placeholder="ví dụ: What is RAG?")
        with col_k:
            top_k = st.number_input("Top-K", 1, 10, 3, key="top_k_search")

        # Metadata filter
        with st.expander("🔧 Metadata filter (tuỳ chọn)"):
            filter_key = st.text_input("Key", placeholder="ví dụ: source")
            filter_val = st.text_input("Value", placeholder="ví dụ: python_intro.txt")

        if st.button("🔎 Search", key="btn_search") and query.strip():
            with st.spinner("Searching…"):
                metadata_filter = {filter_key: filter_val} if filter_key and filter_val else None
                if metadata_filter:
                    results = st.session_state.store.search_with_filter(
                        query, top_k=top_k, metadata_filter=metadata_filter
                    )
                    st.caption(f"🔧 Filter applied: `{metadata_filter}`")
                else:
                    results = st.session_state.store.search(query, top_k=top_k)

            if not results:
                st.info("Không tìm thấy kết quả nào.")
            else:
                for i, r in enumerate(results, 1):
                    src = r["metadata"].get("source", "unknown")
                    score = r["score"]
                    content = r["content"][:280].replace("\n", " ")
                    st.markdown(f"""
                    <div class='chunk-card'>
                        <span class='score-badge'>#{i} · {score:.4f}</span>
                        <span class='source-tag'>{src}</span>
                        <div class='content' style='margin-top:0.5rem;'>{content}…</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: AGENT Q&A (chat)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown("### Agent Q&A (RAG)")

        # LLM selector
        llm_mode = st.radio(
            "LLM",
            ["Demo (Mock)", "Echo Prompt"],
            horizontal=True,
            label_visibility="collapsed",
        )

        def _build_llm(mode: str):
            if mode == "Echo Prompt":
                return lambda p: p
            # Default mock: extract context and format nicely
            def demo_llm(prompt: str) -> str:
                lines = [l for l in prompt.split("\n") if l.startswith("[Chunk")]
                context_summary = " | ".join(l[:80] for l in lines[:3])
                return (
                    f"📚 **Retrieved context:** {context_summary}\n\n"
                    f"💡 *(Demo LLM — kết nối LLM thật để có câu trả lời đầy đủ)*"
                )
            return demo_llm

        # Chat history display
        chat_container = st.container()
        with chat_container:
            for turn in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(turn["q"])
                with st.chat_message("assistant"):
                    st.markdown(turn["a"])

        # Input
        user_q = st.chat_input("Hỏi knowledge base…")
        if user_q:
            from src.agent import KnowledgeBaseAgent
            llm_fn = _build_llm(llm_mode)
            agent = KnowledgeBaseAgent(store=st.session_state.store, llm_fn=llm_fn)

            with st.spinner("Retrieving & generating…"):
                answer = agent.answer(user_q, top_k=3)

            st.session_state.chat_history.append({"q": user_q, "a": answer})

            with chat_container:
                with st.chat_message("user"):
                    st.write(user_q)
                with st.chat_message("assistant"):
                    st.markdown(answer)

        if st.session_state.chat_history:
            if st.button("🗑️ Xoá lịch sử chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: STRATEGY COMPARE
    # ─────────────────────────────────────────────────────────────────────────
    with tab_compare:
        st.markdown("### So Sánh Chunking Strategies")
        compare_text = st.text_area(
            "Text để so sánh (để trống = dùng document đầu tiên)",
            height=120,
            placeholder="Dán văn bản vào đây…",
            key="compare_text",
        )
        cmp_size = st.slider("chunk_size", 50, 500, 200, 25, key="cmp_size")

        if st.button("📊 Compare", key="btn_compare"):
            from src.chunking import ChunkingStrategyComparator
            text_to_use = compare_text.strip()
            if not text_to_use and st.session_state.docs_loaded:
                text_to_use = st.session_state.docs_loaded[0].content[:3000]
                st.caption(f"Dùng: **{st.session_state.docs_loaded[0].id}** (3000 ký tự đầu)")

            if not text_to_use:
                st.warning("Không có text để so sánh.")
            else:
                result = ChunkingStrategyComparator().compare(text_to_use, chunk_size=cmp_size)

                # Stats row
                cols = st.columns(3)
                colors = {"fixed_size": "#38bdf8", "by_sentences": "#a78bfa", "recursive": "#34d399"}
                labels = {"fixed_size": "Fixed Size", "by_sentences": "By Sentences", "recursive": "Recursive"}
                for col, (key, stats) in zip(cols, result.items()):
                    col.markdown(f"""
                    <div class='metric-card' style='border-left:3px solid {colors[key]};'>
                        <div style='font-size:0.7rem;color:{colors[key]};text-transform:uppercase;
                                    letter-spacing:0.08em;margin-bottom:0.3rem;'>{labels[key]}</div>
                        <div style='font-size:1.8rem;font-weight:700;color:#f1f5f9;'>{stats['count']}</div>
                        <div style='font-size:0.75rem;color:#475569;'>chunks</div>
                        <div style='margin-top:0.5rem;font-size:0.85rem;color:#94a3b8;'>
                            avg {stats['avg_length']:.0f} chars</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Preview chunks per strategy
                st.markdown("<div class='section-header' style='margin-top:1.5rem;'>Preview chunks</div>",
                            unsafe_allow_html=True)
                for key, stats in result.items():
                    with st.expander(f"**{labels[key]}** — {stats['count']} chunks"):
                        for i, chunk in enumerate(stats["chunks"][:6], 1):
                            st.markdown(f"""
                            <div class='chunk-card' style='border-left-color:{colors[key]};'>
                                <span class='score-badge' style='background:{colors[key]};'>#{i}</span>
                                <div class='content' style='margin-top:0.4rem;'>{chunk[:200]}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        if len(stats["chunks"]) > 6:
                            st.caption(f"… và {len(stats['chunks'])-6} chunks nữa")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4: DOCUMENTS
    # ─────────────────────────────────────────────────────────────────────────
    with tab_docs:
        st.markdown("### Loaded Documents")
        for doc in st.session_state.docs_loaded:
            size = len(doc.content)
            with st.expander(f"📄 **{doc.id}** — {size:,} chars — `{doc.metadata.get('source', '')}`"):
                st.text(doc.content[:1500] + ("…" if len(doc.content) > 1500 else ""))
                st.caption(f"Metadata: {doc.metadata}")
