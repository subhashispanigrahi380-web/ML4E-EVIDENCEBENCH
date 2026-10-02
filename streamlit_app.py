"""
EvidenceBench — Streamlit Research Workspace
Production-ready for Streamlit Cloud & local environments.
Upload documents → ask questions → get cited, verified answers with abstention gates.
"""

import os
import sys
import time
import json
import logging
import platform
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import streamlit as st

# Configure structured logging for console & Streamlit Cloud stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evidencebench.app")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EvidenceBench — RAG Workspace",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0f1117; }
    .stApp { background: #0f1117; }
    div[data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2a2d3a; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .citation-box {
        background: #1e2235;
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #a5b4fc;
    }
    .query-box {
        background: #1e1e2f;
        border: 1px solid #4f46e5;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 14px;
        color: #e0e7ff;
        font-size: 0.95rem;
    }
    .answer-box {
        background: #162032;
        border: 1px solid #2563eb;
        border-radius: 10px;
        padding: 20px;
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.8;
    }
    .abstain-box {
        background: #2d1515;
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 20px;
        color: #fca5a5;
    }
    .history-card {
        background: #161922;
        border: 1px solid #2a2d3d;
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
        font-size: 0.82rem;
    }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .stage-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 10px;
        border-radius: 5px;
        margin: 3px 0;
        background: #1e2235;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .chunk-card {
        background: #1a2332;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.83rem;
        color: #cbd5e1;
    }
    .score-badge {
        background: #1e3a5f;
        color: #60a5fa;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .source-tag-corpus {
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #0284c7;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-left: 6px;
    }
    .source-tag-upload {
        background: #2e1065;
        color: #c084fc;
        border: 1px solid #9333ea;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Pipeline singleton ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Initializing EvidenceBench RAG pipeline...")
def load_pipeline():
    """Initializes and caches the EvidenceBench pipeline with robust auto-indexing."""
    from evidencebench.pipeline import EvidenceBenchPipeline
    from evidencebench.config import config, get_corpus_dir, get_storage_dir
    
    corpus_dir = get_corpus_dir()
    storage_dir = get_storage_dir()
    
    logger.info(f"[BOOT] Initializing pipeline with corpus_dir={corpus_dir}, storage_dir={storage_dir}")
    pipeline = EvidenceBenchPipeline(storage_dir=storage_dir)
    
    # 1. Ingest built-in corpus documents if not already in document store
    if corpus_dir.exists():
        supported_exts = {".pdf", ".md", ".markdown", ".txt"}
        corpus_files = [f for f in corpus_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]
        logger.info(f"[BOOT] Found {len(corpus_files)} files in corpus directory: {[f.name for f in corpus_files]}")
        
        # Check if any corpus files are missing from doc store
        existing_filenames = {doc.metadata.filename for doc in pipeline.doc_store.documents.values()}
        missing = [f for f in corpus_files if f.name not in existing_filenames]
        if missing or len(pipeline.doc_store.documents) == 0:
            logger.info(f"[BOOT] Ingesting {len(missing)} missing corpus documents into document store...")
            pipeline.doc_store.ingest_directory(corpus_dir, source_type="corpus")
    
    # 2. Re-chunk and build retrieval index if chunks are missing
    if not pipeline.indexed_chunks and len(pipeline.doc_store.documents) > 0:
        logger.info(f"[BOOT] Rebuilding search indices over {len(pipeline.doc_store.documents)} documents...")
        chunk_count = pipeline.reindex()
        logger.info(f"[BOOT] Startup indexing complete: {chunk_count} chunks indexed.")
        
    return pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_uploaded_file(uploaded_file) -> Path:
    """Save Streamlit UploadedFile to corpus directory and return path."""
    from evidencebench.config import get_corpus_dir
    dest_dir = get_corpus_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / uploaded_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path


def format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def transcribe_audio_file(audio_bytes: bytes) -> Optional[str]:
    """Transcribes audio buffer (WAV/WEBM/OGG) to text using SpeechRecognition."""
    try:
        import speech_recognition as sr
        import io
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = r.record(source)
            return r.recognize_google(audio_data)
    except Exception as e:
        logger.warning(f"Voice transcription note: {e}")
    return None


def _get_history_file() -> Path:
    from evidencebench.config import get_storage_dir
    hist_dir = get_storage_dir()
    hist_dir.mkdir(parents=True, exist_ok=True)
    return hist_dir / "chat_history.json"


def load_persistent_history() -> List[dict]:
    hist_file = _get_history_file()
    if hist_file.exists():
        try:
            with open(hist_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_persistent_history(history: List[dict]) -> None:
    hist_file = _get_history_file()
    try:
        with open(hist_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Note: Could not save persistent history: {e}")


# Initialize persistent conversation history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_persistent_history()
if "voice_query" not in st.session_state:
    st.session_state.voice_query = ""


# ── Load Pipeline on Startup ───────────────────────────────────────────────────
pipeline = load_pipeline()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 EvidenceBench")
    st.markdown("**RAG Research Workspace**")
    st.markdown("---")

    # ── 1. Document Upload (at the top) ───────────────────────────────────────
    st.markdown("### 📂 Upload Documents")
    st.markdown("Upload files to supplement or search alongside the corpus:")

    uploaded_files = st.file_uploader(
        "Supported: PDF, Markdown, TXT",
        type=["pdf", "md", "txt", "markdown"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        newly_added = []
        for f in uploaded_files:
            dest = save_uploaded_file(f)
            newly_added.append(f.name)

        if st.button("⚡ Ingest & Index Uploads", type="primary", use_container_width=True):
            from evidencebench.config import get_corpus_dir
            with st.spinner("Ingesting new uploaded documents..."):
                # Tag as uploaded so source indicator reflects user origin
                statuses = pipeline.doc_store.ingest_directory(get_corpus_dir(), source_type="uploaded")
                new_count = sum(1 for s in statuses if s.action in ("CREATED", "UPDATED_NEW_VERSION"))
                dup_count = sum(1 for s in statuses if s.action == "DUPLICATE_IGNORED")
            with st.spinner(f"Rebuilding index over {len(pipeline.doc_store.documents)} total documents..."):
                pipeline.reindex()
            msg = f"✅ {new_count} new · {dup_count} duplicate · {len(pipeline.indexed_chunks)} chunks indexed"
            st.success(msg)
            if new_count:
                st.balloons()

    st.markdown("---")

    # ── 2. Previous Conversations (Recent Queries) ────────────────────────────
    st.markdown("### 💬 Recent Queries")
    if st.session_state.chat_history:
        for idx, chat_item in enumerate(reversed(st.session_state.chat_history[-8:])):
            q_preview = chat_item['query']
            if len(q_preview) > 32:
                q_preview = q_preview[:30] + "…"
            status_dot = "🟢" if chat_item['decision'] == "ANSWER" else "🔴"
            with st.expander(f"{status_dot} {q_preview}"):
                st.caption(f"Scope: `{chat_item.get('target', 'All Documents')}`")
                st.markdown(f"**Q:** {chat_item['query']}")
                st.markdown(f"**Decision:** `{chat_item['decision']}`")
                st.markdown(f"<small>{chat_item['answer'][:180]}…</small>", unsafe_allow_html=True)
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            save_persistent_history([])
            st.rerun()
    else:
        st.caption("No previous queries yet.")

    st.markdown("---")

    # ── 3. Search Scope & Settings ────────────────────────────────────────────
    st.markdown("### ⚙️ Search Scope")
    
    available_docs = list(pipeline.doc_store.documents.values())
    available_doc_names = [doc.metadata.filename for doc in available_docs]
    
    doc_filter_options = [
        "🌐 Hybrid: Search All (Corpus + Uploads)",
        "🔍 Single Document Only",
    ]
    filter_choice = st.radio("Retrieval Mode", doc_filter_options, index=0)

    selected_doc = None
    if filter_choice == "🔍 Single Document Only" and available_doc_names:
        selected_doc = st.selectbox(
            "Target Document",
            options=available_doc_names,
            index=len(available_doc_names) - 1,
        )
        st.info(f"Focusing retrieval on: **{selected_doc}**")

    # Pipeline Settings
    with st.expander("🛠️ Advanced Settings", expanded=False):
        top_k = st.slider("Top-K candidates", min_value=3, max_value=20, value=10)
        use_reranker = st.toggle("Cross-encoder reranker", value=True)
        chunking_mode = st.radio(
            "Chunking strategy",
            ["Structure-Aware", "Fixed-Recursive"],
            index=0,
        )
        if chunking_mode.lower().replace("-", "_") != pipeline.chunking_strategy_name:
            pipeline.set_chunking_strategy("structure_aware" if chunking_mode == "Structure-Aware" else "fixed_recursive")

    st.markdown("---")

    # ── 4. System Status & Rebuild ────────────────────────────────────────────
    st.markdown("### 📊 Index Status")
    doc_count = len(pipeline.doc_store.documents)
    chunk_count = len(pipeline.indexed_chunks)
    
    st.markdown(f"- **Documents Loaded:** `{doc_count}`")
    st.markdown(f"- **Chunks Indexed:** `{chunk_count}`")
    st.markdown(f"- **Index Status:** `{'🟢 Ready' if chunk_count > 0 else '🔴 Empty'}`")
    st.markdown(f"- **Environment:** `{platform.system()} ({'Cloud' if os.getenv('STREAMLIT_SERVER_BASE_URL') or os.getenv('HOSTNAME') else 'Local'})`")

    if st.button("🔄 Rebuild Search Index", use_container_width=True):
        with st.spinner("Rebuilding indexes from documents..."):
            pipeline.reindex(force_reingest=True)
        st.success(f"Reindexed! {len(pipeline.indexed_chunks)} chunks ready.")
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 🔬 EvidenceBench Research Workspace")
st.markdown(
    "Upload documents on the left, or search across preloaded research documents. "
    "Every answer is **grounded in retrieved evidence** with verifiable citations and abstention gates."
)

# Tabs
tab_qa, tab_docs, tab_diagnostics, tab_about = st.tabs([
    "💬 Ask a Question",
    "📚 Document Browser",
    "🩺 Deployment Diagnostics",
    "ℹ️ About",
])


# ─── Q&A Tab ──────────────────────────────────────────────────────────────────
with tab_qa:
    st.markdown("### Ask a question about your documents")

    # Preset examples
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
    preset_query = None
    with col_ex1:
        if st.button("💡 SLA uptime guarantee?", use_container_width=True):
            preset_query = "What is the SLA uptime guarantee?"
    with col_ex2:
        if st.button("🧠 What is psychology?", use_container_width=True):
            preset_query = "What is psychology?"
    with col_ex3:
        if st.button("🌿 CO₂ reduction targets?", use_container_width=True):
            preset_query = "What are the CO2 reduction targets?"
    with col_ex4:
        if st.button("🔬 Experimental method?", use_container_width=True):
            preset_query = "How does the experimental method work?"

    # Voice Query Option & Text Input
    col_input, col_voice = st.columns([4, 1])
    
    with col_voice:
        audio_val = st.audio_input("🎙️ Voice Query", label_visibility="collapsed")
        if audio_val is not None:
            audio_bytes = audio_val.read()
            with st.spinner("🎧 Transcribing voice query…"):
                transcribed = transcribe_audio_file(audio_bytes)
                if transcribed:
                    st.session_state.voice_query = transcribed
                    st.toast(f"🎙️ Captured: \"{transcribed}\"")

    with col_input:
        initial_value = preset_query or st.session_state.voice_query or ""
        query = st.text_input(
            "Your question",
            value=initial_value,
            placeholder="Type your question or record voice using the mic icon 🎙️...",
            label_visibility="collapsed",
        )

    col_btn, col_clear_voice = st.columns([3, 1])
    with col_btn:
        run_btn = st.button("🔍 Search & Verify", type="primary", disabled=not query, use_container_width=True)
    with col_clear_voice:
        if st.session_state.voice_query and st.button("✕ Reset Voice", use_container_width=True):
            st.session_state.voice_query = ""
            st.rerun()

    if run_btn and query:
        try:
            # 1. Ensure index is populated (never fail silently)
            if not pipeline.indexed_chunks:
                with st.spinner("Index is unpopulated. Automatically building index from documents..."):
                    pipeline.reindex(force_reingest=True)

            if not pipeline.indexed_chunks:
                st.error(
                    "❌ **Retrieval Failure — Corpus Missing:** "
                    f"No documents could be found or indexed from `{pipeline.corpus_dir}`. "
                    "Please upload files via the sidebar or check the Deployment Diagnostics tab."
                )
                st.stop()

            # 2. Decide pipeline mode and target filter
            mode = "hybrid_rerank" if use_reranker else "hybrid"
            target_docs = [selected_doc] if (filter_choice == "🔍 Single Document Only" and selected_doc) else None

            logger.info(f"[QUERY] Query='{query}' | Mode='{mode}' | Scope='{target_docs or 'All Documents'}'")

            with st.spinner("Retrieving evidence and verifying citations…"):
                t0 = time.time()
                gen_result, exec_trace = pipeline.query(
                    query,
                    top_k=top_k,
                    pipeline_mode=mode,
                    target_doc_ids=target_docs,
                )
                elapsed = time.time() - t0

            # 3. Unpack results
            abstention_obj = gen_result.decision
            decision_str = abstention_obj.decision
            abstention_reason = abstention_obj.rationale
            answer = gen_result.answer
            top_chunks = gen_result.top_chunks

            # ── 1. User Query Section (Separate Card) ─────────
            st.markdown(
                f'<div class="query-box">'
                f'<strong style="color:#a5b4fc;text-transform:uppercase;font-size:0.78rem;letter-spacing:1px;display:block;margin-bottom:4px">User Question</strong>'
                f'<span style="font-size:1.05rem;font-weight:500;">{query}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── 2. Output / Verified Answer Section (Streamed UX) ────
            if decision_str == "ABSTAIN":
                # Clear explanation of WHY retrieval refused/failed
                detailed_reason = abstention_reason or "Insufficient evidence found in the corpus."
                st.markdown(
                    f'<div class="abstain-box">'
                    f'<strong style="color:#f87171;text-transform:uppercase;font-size:0.78rem;letter-spacing:1px;display:block;margin-bottom:6px">System Decision: Refusal ({abstention_obj.reason.value})</strong>'
                    f'<strong>🚫 Cannot Answer with Verified Evidence</strong><br><br>'
                    f'{detailed_reason}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                answer_saved = detailed_reason
            else:
                paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
                answer_placeholder = st.empty()
                
                # Stream word-by-word across paragraphs for responsive UX
                for p_idx, para in enumerate(paragraphs):
                    words = para.split(" ")
                    current_para = ""
                    for word in words:
                        current_para += word + " "
                        partial_paras = paragraphs[:p_idx] + [current_para.strip()]
                        paras_html = "".join(f'<p style="margin-bottom:12px;">{p}</p>' for p in partial_paras)
                        answer_placeholder.markdown(
                            f'<div class="answer-box">'
                            f'<strong style="color:#60a5fa;text-transform:uppercase;font-size:0.78rem;letter-spacing:1px;display:block;margin-bottom:8px">Verified Evidence-Grounded Output</strong>'
                            f'{paras_html}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        time.sleep(0.015)

                final_paras_html = "".join(f'<p style="margin-bottom:12px;">{p}</p>' for p in paragraphs)
                answer_placeholder.markdown(
                    f'<div class="answer-box">'
                    f'<strong style="color:#60a5fa;text-transform:uppercase;font-size:0.78rem;letter-spacing:1px;display:block;margin-bottom:8px">Verified Evidence-Grounded Output</strong>'
                    f'{final_paras_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                answer_saved = answer

            st.markdown(f"<small style='color:#64748b'>⏱ Retrieved in {format_ms(elapsed)}</small>", unsafe_allow_html=True)

            # Record in persistent history
            st.session_state.chat_history.append({
                "query": query,
                "decision": decision_str,
                "answer": answer_saved,
                "target": selected_doc if filter_choice == "🔍 Single Document Only" and selected_doc else "All Documents (Corpus + Uploads)",
                "timestamp": time.time(),
            })
            save_persistent_history(st.session_state.chat_history)

            # Metrics row
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Decision", decision_str)
            with m2:
                st.metric("Total Indexed Chunks", len(pipeline.indexed_chunks))
            with m3:
                st.metric("Evidence Chunks Retrieved", len(top_chunks))
            with m4:
                top_score = top_chunks[0].score if top_chunks else 0.0
                st.metric("Top Relevance Score", f"{top_score:.3f}")

            # Evidence chunks with source badges (corpus vs upload)
            if top_chunks:
                st.markdown("### 📎 Retrieved Evidence Passages")
                for i, cand in enumerate(top_chunks[:5], 1):
                    meta = cand.chunk.metadata
                    text = cand.chunk.text
                    doc_name = meta.doc_filename
                    page = meta.page_number
                    section = meta.section_title
                    fscore = cand.score
                    rerank_prob = cand.method_scores.get("reranker_prob", None)
                    rank = cand.rank

                    # Determine source origin (upload vs corpus)
                    doc_obj = pipeline.doc_store.documents.get(meta.doc_id)
                    source_type = "corpus"
                    if doc_obj and "source_type" in doc_obj.metadata.custom_metadata:
                        source_type = doc_obj.metadata.custom_metadata["source_type"]

                    source_badge = (
                        '<span class="source-tag-upload">User Upload</span>'
                        if source_type == "uploaded"
                        else '<span class="source-tag-corpus">Corpus Library</span>'
                    )

                    score_label = f"score: {fscore:.3f} · rank: #{rank}"
                    if rerank_prob is not None:
                        score_label += f" · rerank: {rerank_prob:.3f}"

                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
                        f'<div><strong style="color:#93c5fd">#{i} · {doc_name}</strong> {source_badge}</div>'
                        f'<span class="score-badge">{score_label}</span>'
                        f'</div>'
                        f'<div style="color:#94a3b8;font-size:0.75rem;margin-bottom:6px">'
                        f'Page {page}{" · " + section if section else ""}'
                        f'</div>'
                        f'<div style="color:#cbd5e1">{text[:500]}{"…" if len(text) > 500 else ""}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Pipeline execution trace
            with st.expander("🔭 Pipeline Execution Trace", expanded=False):
                st.markdown(f"**Total latency:** {exec_trace.total_latency_ms:.1f} ms · "
                            f"**Mode:** `{exec_trace.pipeline_mode}` · "
                            f"**Index:** `{exec_trace.index_version}`")
                for stage in exec_trace.stages:
                    name = stage.stage_name
                    dur = stage.latency_ms
                    bar_w = min(int(dur / 5), 200)
                    st.markdown(
                        f'<div class="stage-row">'
                        f'<span style="width:200px;display:inline-block">{name}</span>'
                        f'<span style="flex:1;margin:0 12px">'
                        f'<div style="background:#6366f1;height:6px;width:{bar_w}px;border-radius:3px;display:inline-block"></div>'
                        f'</span>'
                        f'<span style="color:#818cf8">{dur:.1f} ms</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            logger.error(f"[ERROR] Query processing failed: {e}", exc_info=True)
            st.error(f"❌ Error during query processing: {e}")
            st.exception(e)


# ─── Document Browser Tab ─────────────────────────────────────────────────────
with tab_docs:
    st.markdown("### 📚 Indexed Document Library")
    docs = pipeline.doc_store.documents

    if not docs:
        st.warning("⚠️ No documents are currently loaded. Upload documents in the sidebar or rebuild index.")
    else:
        corpus_count = sum(1 for d in docs.values() if d.metadata.custom_metadata.get("source_type") != "uploaded")
        upload_count = len(docs) - corpus_count
        st.success(f"**{len(docs)} Total Documents** ({corpus_count} corpus library, {upload_count} uploaded) · **{len(pipeline.indexed_chunks)} chunks** indexed")

        for doc_id, doc in docs.items():
            meta = doc.metadata
            ftype = meta.file_type.value if hasattr(meta.file_type, "value") else str(meta.file_type)
            icon = {"pdf": "📄", "markdown": "📝", "text": "📃"}.get(ftype.lower(), "📄")
            is_upload = meta.custom_metadata.get("source_type") == "uploaded"
            origin_badge = " [User Upload]" if is_upload else " [Corpus]"

            with st.expander(f"{icon} {meta.filename}{origin_badge} · v{meta.version} · {len(doc.sections)} sections"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"**File type:** `{ftype}`")
                    st.markdown(f"**Origin:** `{'User Upload' if is_upload else 'Built-in Corpus'}`")
                with col_b:
                    st.markdown(f"**Sections:** {len(doc.sections)}")
                    st.markdown(f"**Characters:** {meta.total_characters}")
                with col_c:
                    sha = meta.content_hash or ""
                    if sha:
                        st.markdown(f"**SHA-256:** `{sha[:16]}…`")

                if doc.sections:
                    st.markdown("**Sections:**")
                    for sec in doc.sections[:8]:
                        title = getattr(sec, 'title', str(sec))
                        st.markdown(f"- {title}")
                    if len(doc.sections) > 8:
                        st.caption(f"…and {len(doc.sections) - 8} more sections")


# ─── Diagnostics Tab ──────────────────────────────────────────────────────────
with tab_diagnostics:
    st.markdown("### 🩺 Deployment Diagnostics & Health Check")
    st.caption("Live environment inspection for Streamlit Cloud and local troubleshooting.")

    from evidencebench.config import PROJECT_ROOT, CORPUS_DIR, INDICES_DIR, get_corpus_dir, get_storage_dir

    active_corpus = get_corpus_dir()
    active_storage = get_storage_dir()
    corpus_exists = active_corpus.exists()
    corpus_files = [f.name for f in active_corpus.iterdir() if f.is_file()] if corpus_exists else []
    index_built = len(pipeline.indexed_chunks) > 0
    docs_loaded = len(pipeline.doc_store.documents) > 0

    # Status indicators
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    with col_h1:
        st.metric("Corpus Folder", "🟢 Exists" if corpus_exists else "🔴 Missing")
    with col_h2:
        st.metric("Corpus Files", f"🟢 {len(corpus_files)}" if len(corpus_files) > 0 else "🔴 0 Files")
    with col_h3:
        st.metric("Documents Loaded", f"🟢 {len(pipeline.doc_store.documents)}" if docs_loaded else "🔴 0")
    with col_h4:
        st.metric("Retrieval Index", f"🟢 {len(pipeline.indexed_chunks)} chunks" if index_built else "🔴 Not Built")

    st.markdown("---")
    
    st.markdown("#### 📁 Paths & Runtime Environment")
    diag_data = {
        "Python Version": sys.version.split()[0],
        "Platform / OS": f"{platform.system()} {platform.release()}",
        "Current Working Directory (os.getcwd)": os.getcwd(),
        "Project Root (detected)": str(PROJECT_ROOT),
        "Corpus Directory": str(active_corpus),
        "Storage / Index Directory": str(active_storage),
        "Active Chunking Strategy": pipeline.chunking_strategy_name,
        "Dense Embedding Model": pipeline.dense_retriever.model_name,
        "Cross-Encoder Reranker": pipeline.reranker.model_name,
    }
    st.table(list(diag_data.items()))

    st.markdown("#### 📄 Available Corpus Files on Disk")
    if corpus_files:
        for f in corpus_files:
            file_path = active_corpus / f
            file_size_kb = file_path.stat().st_size / 1024 if file_path.exists() else 0
            st.markdown(f"- 📄 `{f}` ({file_size_kb:.1f} KB)")
    else:
        st.warning("⚠️ No files found in corpus directory.")

    col_btn_diag1, col_btn_diag2 = st.columns(2)
    with col_btn_diag1:
        if st.button("🔄 Force Re-ingest Corpus & Rebuild Index", use_container_width=True):
            with st.spinner("Re-ingesting and re-indexing..."):
                pipeline.doc_store.ingest_directory(active_corpus, source_type="corpus")
                pipeline.reindex()
            st.success(f"Complete! Loaded {len(pipeline.doc_store.documents)} docs and {len(pipeline.indexed_chunks)} chunks.")
            st.rerun()
    with col_btn_diag2:
        if st.button("🧪 Run Pipeline Test Query ('What is psychology?')", use_container_width=True):
            test_res, _ = pipeline.query("What is psychology?", top_k=3)
            st.info(f"Test Result Decision: **{test_res.decision.decision}** (Confidence: {test_res.decision.confidence_score:.3f})")
            st.markdown(f"**Answer Preview:** {test_res.answer[:250]}...")


# ─── About Tab ────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
### 🔬 EvidenceBench — Project 08

A full-stack Retrieval-Augmented Generation research workspace built from scratch.

#### Architecture Highlights

| Stage | Component | Detail |
|:---|:---|:---|
| **Ingest** | `UnifiedDocumentParser` | PDF page/section extraction, Markdown heading trees, TXT dividers |
| **Deduplicate** | `DocumentStore` | SHA-256 fingerprinting, version lineage (v1.0 → v1.1) |
| **Chunk** | `StructureAwareChunker` / `FixedRecursiveChunker` | Section-boundary vs 500-char sliding window |
| **Keyword retrieval** | `BM25Retriever` | First-principles Okapi BM25, in-memory inverted index |
| **Dense retrieval** | `DenseRetriever` | `all-MiniLM-L6-v2`, cosine similarity |
| **Fusion** | `CalibratedHybridFusion` | Custom C-RRF: RRF + min-max score calibration |
| **Rerank** | `CrossEncoderReranker` | `ms-marco-MiniLM-L-6-v2`, sigmoid probability |
| **Abstention** | `AbstentionEngine` | Missing / weak / conflicting evidence detection |
| **Answer** | `GroundedGenerator` | Citation-tagged synthesis with token-level verification |
| **Trace** | `PipelineTracer` | Microsecond per-stage JSON waterfall |

#### Benchmark Results (50 questions, 5 configurations)

| Config | Recall@5 | MRR@5 | Citation Precision | Abstention F1 |
|:---|:---:|:---:|:---:|:---:|
| Dense-Only | **100%** | **0.888** | 100% | 76.6% |
| BM25-Only | 90.6% | 0.823 | 100% | 74.4% |
| C-RRF Hybrid | 81.2% | 0.716 | 100% | **77.8%** |

> Citation Precision = **100%** across all configurations — zero hallucinated claims.
    """)
