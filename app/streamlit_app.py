"""Streamlit web UI for the 3GPP KG-RAG Chatbot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.pipeline.rag_pipeline import KGRAGPipeline, PipelineResponse

st.set_page_config(
    page_title="3GPP KG-RAG Chatbot",
    page_icon="📡",
    layout="wide",
)

# ── Custom CSS for badges ──────────────────────────────────────────
st.markdown("""
<style>
.badge-pass  { background:#d4edda; color:#155724; padding:4px 12px;
               border-radius:12px; font-size:0.85em; font-weight:600;
               border:1px solid #c3e6cb; display:inline-block; margin:2px 4px; }
.badge-fail  { background:#f8d7da; color:#721c24; padding:4px 12px;
               border-radius:12px; font-size:0.85em; font-weight:600;
               border:1px solid #f5c6cb; display:inline-block; margin:2px 4px; }
.badge-warn  { background:#fff3cd; color:#856404; padding:4px 12px;
               border-radius:12px; font-size:0.85em; font-weight:600;
               border:1px solid #ffeeba; display:inline-block; margin:2px 4px; }
.badge-info  { background:#d1ecf1; color:#0c5460; padding:4px 12px;
               border-radius:12px; font-size:0.85em; font-weight:600;
               border:1px solid #bee5eb; display:inline-block; margin:2px 4px; }
.conf-high   { color:#28a745; font-weight:700; }
.conf-med    { color:#ffc107; font-weight:700; }
.conf-low    { color:#dc3545; font-weight:700; }
</style>
""", unsafe_allow_html=True)


def _badge(label: str, status: str) -> str:
    css = {"pass": "badge-pass", "fail": "badge-fail", "warn": "badge-warn", "info": "badge-info"}
    return f'<span class="{css.get(status, "badge-info")}">{label}</span>'


def _confidence_label(score: float) -> tuple[str, str]:
    if score >= 0.75:
        return "High", "conf-high"
    if score >= 0.50:
        return "Medium", "conf-med"
    return "Low", "conf-low"


def render_verification_badges(response: PipelineResponse) -> None:
    """Render verification status badges."""
    badges = []

    # Overall status
    if response.rejected:
        badges.append(_badge("Answer Rejected", "fail"))
    else:
        badges.append(_badge("Answer Approved", "pass"))

    # Chain-of-Noting
    if response.noting_result:
        if response.noting_result.can_answer:
            badges.append(_badge(f"CoN Pass ({response.noting_result.confidence:.0%})", "pass"))
        else:
            badges.append(_badge("CoN Failed", "fail"))

    # Citations
    if response.citation_result:
        invalid = len(response.citation_result.invalid_citations)
        valid = len(response.citation_result.valid_citations)
        if invalid == 0 and valid > 0:
            badges.append(_badge(f"Citations Verified ({valid})", "pass"))
        elif invalid > 0:
            badges.append(_badge(f"Citation Issues ({invalid})", "fail"))
        else:
            badges.append(_badge("No Citations", "warn"))

    # Grounding
    if response.grounding_result:
        if response.grounding_result.grounded:
            badges.append(_badge(f"Grounded ({response.grounding_result.score:.0%})", "pass"))
        else:
            badges.append(_badge("Ungrounded Claims", "fail"))

    # Reflection agent
    if response.verification_result:
        verdict = response.verification_result.verdict
        conf = response.verification_result.confidence
        if verdict == "APPROVED":
            badges.append(_badge(f"Reflection Approved ({conf:.0%})", "pass"))
        elif verdict == "REJECTED":
            badges.append(_badge("Reflection Rejected", "fail"))
        else:
            badges.append(_badge(f"Needs Revision ({conf:.0%})", "warn"))

    # Generation mode
    badges.append(_badge(f"Mode: {response.generation_mode.title()}", "info"))

    st.markdown(" ".join(badges), unsafe_allow_html=True)


def render_confidence_score(response: PipelineResponse) -> None:
    """Display confidence score with progress bar."""
    score = response.confidence_score
    label, css_class = _confidence_label(score)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.progress(min(score, 1.0), text=f"Confidence: {score:.0%} ({label})")
    with col2:
        st.markdown(
            f'<p class="{css_class}">Score: {score:.0%}</p>',
            unsafe_allow_html=True,
        )
    with col3:
        st.caption(f"{len(response.spec_chunks)} spec chunks · {len(response.retrieved_triples)} KG triples")


def render_answer(response: PipelineResponse) -> None:
    """Render the answer with clean markdown formatting."""
    if response.rejected:
        st.markdown("---")
        st.markdown("### Unable to Answer")
        st.info(response.answer)
        if response.rejection_reason:
            st.markdown(f"**Reason:** {response.rejection_reason}")
        return

    st.markdown("---")
    st.markdown(response.answer)


def build_debug_info(response: PipelineResponse) -> dict:
    return {
        "generation_mode": response.generation_mode,
        "confidence_score": response.confidence_score,
        "route": {
            "intent": response.route.intent.value if response.route else None,
            "index_hint": response.route.index_hint if response.route else None,
            "keywords": response.route.keywords if response.route else [],
        },
        "primary_spec_chunks": len(response.spec_chunks),
        "kg_triples": len(response.retrieved_triples),
        "noting": {
            "can_answer": response.noting_result.can_answer if response.noting_result else None,
            "confidence": response.noting_result.confidence if response.noting_result else None,
        },
        "verification": {
            "verdict": response.verification_result.verdict if response.verification_result else None,
            "issues": response.verification_result.issues if response.verification_result else [],
        },
        "citations": {
            "valid": len(response.citation_result.valid_citations) if response.citation_result else 0,
            "invalid": len(response.citation_result.invalid_citations) if response.citation_result else 0,
        },
        "grounding": {
            "grounded": response.grounding_result.grounded if response.grounding_result else None,
            "score": response.grounding_result.score if response.grounding_result else None,
        },
    }


# ── Page layout ────────────────────────────────────────────────────
st.title("📡 3GPP KG-RAG Chatbot")
st.caption(
    "Retrieval-Augmented Generation for 3GPP telecom standards · "
    "Primary source: [GSMA/telecom-kg-rel19](https://huggingface.co/datasets/GSMA/telecom-kg-rel19) · "
    "Near-zero hallucination guardrails"
)

with st.sidebar:
    st.header("About")
    st.markdown("""
    **Primary source:** [GSMA/telecom-kg-rel19](https://huggingface.co/datasets/GSMA/telecom-kg-rel19)
    on Hugging Face (3GPP Rel-19 mirror)

    **Pipeline:**
    1. Retrieve from HF 3GPP corpus
    2. KG triples as supplement
    3. Chain-of-Noting check
    4. Extractive generation (verbatim quotes)
    5. Citation + grounding verification
    6. Reflection agent approval
    """)
    st.header("Sample Questions")
    sample_questions = [
        "What is RRC connection establishment?",
        "What are the functions of the AMF?",
        "What is the purpose of SIB1?",
        "How does network slicing work in 5G?",
        "What is the T300 timer used for?",
        "What is the difference between CM-IDLE and CM-CONNECTED?",
    ]
    for q in sample_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["query_input"] = q

    st.header("Settings")
    show_debug = st.checkbox("Show debug info", value=False)


@st.cache_resource
def load_pipeline():
    pipeline = KGRAGPipeline()
    pipeline.initialize()
    return pipeline


try:
    pipeline = load_pipeline()
    st.success("Pipeline loaded and ready")
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.info("Run `python scripts/build_index.py` first to build the search index.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("response_meta"):
            render_verification_badges(msg["response_meta"])
            render_confidence_score(msg["response_meta"])
        st.markdown(msg["content"])
        if show_debug and msg.get("debug"):
            with st.expander("Debug Info"):
                st.json(msg["debug"])

query = st.chat_input("Ask a question about 3GPP standards...")
if not query and "query_input" in st.session_state:
    query = st.session_state.pop("query_input")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and verifying..."):
            response = pipeline.query(query)

        render_verification_badges(response)
        render_confidence_score(response)
        render_answer(response)

        debug_info = build_debug_info(response) if show_debug else None

        if show_debug:
            with st.expander("Debug Info"):
                st.json(debug_info)

            if response.retrieved_chunks:
                with st.expander("Retrieved Chunks"):
                    for c in response.retrieved_chunks[:5]:
                        st.markdown(f"**{c.citation()}** · score `{c.score:.3f}`")
                        st.markdown(f"> {c.text[:350]}...")

            if response.retrieved_triples:
                with st.expander("KG Triples"):
                    for t in response.retrieved_triples[:5]:
                        st.markdown(f"- `{t.to_text()[:200]}`")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer if not response.rejected else response.answer,
            "response_meta": response,
            "debug": debug_info,
        })
