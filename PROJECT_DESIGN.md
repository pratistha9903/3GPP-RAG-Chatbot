# 3GPP KG-RAG Chatbot — Project Design Document

**Author:** Pratistha Srivastava  
**Project:** RAG-based Chatbot for 3GPP Telecom Standards  
**Goal:** Near-zero hallucinations using 3GPP documentation as the primary knowledge source

---

## 1. Problem Statement

Telecom engineers rely on **3GPP standards** (TS documents) for accurate technical information. General-purpose LLM chatbots often **hallucinate** — inventing spec numbers, procedures, or parameters that do not exist.

This project builds a **Retrieval-Augmented Generation (RAG)** chatbot that:
- Uses **official 3GPP Release 19 documentation** as the primary knowledge source
- Returns **verbatim, citation-backed answers** from retrieved spec text
- **Rejects** questions when evidence is insufficient rather than guessing

---

## 2. Design Goals

| Goal | Design choice |
|------|----------------|
| Near-zero hallucinations | Extractive-only generation (no external LLM) |
| 3GPP as primary source | Hugging Face `GSMA/telecom-kg-rel19` dataset |
| Accurate retrieval | Hybrid BM25 + semantic search with RRF fusion |
| Relationship queries | Knowledge graph triple supplement |
| Trustworthy output | Multi-layer verification before answer release |
| Runnable without API keys | Fully local pipeline |

---

## 3. Primary Knowledge Source

**Dataset:** [GSMA/telecom-kg-rel19](https://huggingface.co/datasets/GSMA/telecom-kg-rel19)

| Asset | Description |
|-------|-------------|
| `rel19_3gpp_telecom_kg.graphml` | Knowledge graph (21K+ nodes, 31K+ edges) |
| `rel19_text_chunks.jsonl` | Chunk metadata (896K IDs with TS/release provenance) |
| `entities.json` | Entity name mappings |

**3GPP text reconstruction:** The JSONL chunk file contains metadata only (empty `text` fields). Primary spec text is reconstructed by grouping KG-grounded node descriptions by `chunk_id`, yielding ~8,032 authoritative HF spec chunks.

**Index composition after build:**
- 8,032 — Primary HF 3GPP spec chunks
- 14,072 — KG supplement chunks
- 22,104 — Total indexed chunks

---

## 4. System Architecture

```
┌─────────────┐
│  User Query │
└──────┬──────┘
       ▼
┌──────────────────┐
│  Query Router     │  Intent classification (definition, procedure, etc.)
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌─────────────┐
│ Hybrid │ │ KG Triple   │
│ Search │ │ Retriever   │  ← Query expansion happens inside Hybrid Search
└───┬────┘ └──────┬──────┘
    └──────┬──────┘
           ▼
┌──────────────────┐
│ Require 3GPP      │  Reject if no primary spec chunks retrieved
│ Evidence Check    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Chain-of-Noting   │  Context sufficiency check (MockLLM + retrieval rules)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Extractive Gen.   │  Verbatim quotes from retrieved chunks
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Citation Verify   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Grounding Verify  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Reflection Agent  │
└────────┬─────────┘
         ▼
    Final Answer
    (or Rejection)
```

---

## 5. Component Design

### 5.1 Indexing Layer (`src/indexing/`)

| Module | Responsibility |
|--------|----------------|
| `hf_loader.py` | Downloads and loads HF dataset; reconstructs 3GPP text from KG descriptions |
| `chunker.py` | Hierarchical chunking preserving clause/section structure |
| `embedder.py` | Sentence-transformer embeddings (`all-MiniLM-L6-v2`, 384-dim) |
| `indexer.py` | Builds and persists BM25 + vector index |
| `spec_loader.py` | Source type constants; KG supplement chunk loader |

### 5.2 Retrieval Layer (`src/retrieval/`)

| Module | Responsibility |
|--------|----------------|
| `hybrid_retriever.py` | BM25 + cosine similarity search, RRF fusion, **query expansion**, 3× spec boost, parent-context enrichment |
| `query_router.py` | Classifies intent (definition, procedure, relationship, etc.) — does **not** expand queries |
| `kg_retriever.py` | Retrieves ontology triples for relationship queries |

**Query expansion** (in `hybrid_retriever.py`, not the query router):
- "phone" → UE, mobile device
- "connect" → RRC, attach, register, camp
- "network" → cell, RAN, PLMN, network selection

### 5.3 Generation Layer (`src/generation/`)

| Module | Responsibility |
|--------|----------------|
| `extractive_generator.py` | **Active generator** — selects most relevant verbatim excerpts; builds cited answer |
| `llm_client.py` | `MockLLM` — rule-based agent backing CoN and reflection (no external API) |
| `reflection_agent.py` | **Active** — final cross-validation of answer vs retrieved context |
| `prompts.py` | Prompt templates used by CoN and reflection agents |
| `compliance_agent.py` | Present in codebase but **not used** in the current extractive pipeline |

**Extractive generation** eliminates generative hallucination at the source — the system never paraphrases or invents text.

### 5.4 Verification Layer (`src/verification/`)

| Module | Responsibility |
|--------|----------------|
| `chain_of_noting.py` | **Active** — evaluates whether retrieved context is sufficient; uses `MockLLM` + retrieval-confidence rules |
| `citation_verifier.py` | **Active** — rejects citations (TS/clause) not present in retrieved chunks |
| `grounding_verifier.py` | **Active** — rejects claims not supported by retrieved text |

### 5.5 Knowledge Graph (`src/knowledge_graph/`)

| Module | Responsibility |
|--------|----------------|
| `loader.py` | Loads GraphML from HF dataset; NetworkX `DiGraph`; node/edge indexing |

### 5.6 Pipeline (`src/pipeline/rag_pipeline.py`)

Orchestrates the full flow in this **exact order** (see `KGRAGPipeline.query()`):

1. `SemanticQueryRouter.route()` — classify intent
2. `HybridRetriever.retrieve_primary_specs()` — hybrid search + query expansion
3. `HybridRetriever.enrich_with_parent_context()` — add parent section text
4. `KGRetriever.retrieve_triples()` — KG supplement (if enabled)
5. **Require primary 3GPP evidence** — reject if `spec_chunks` below minimum
6. `ChainOfNoting.evaluate()` — context sufficiency check
7. `ExtractiveAnswerBuilder.build()` — verbatim quoted answer
8. `CitationVerifier.verify()` — citation validation
9. `GroundingVerifier.verify()` — claim grounding check
10. `ReflectionAgent.verify()` — final approval/rejection

---

## 6. Zero-Hallucination Strategy

Six layers of protection:

1. **Require primary 3GPP evidence** — no spec chunks → reject
2. **Chain-of-Noting** — insufficient context → reject
3. **Extractive generation** — verbatim quotes only, no LLM paraphrasing
4. **Citation verification** — invalid TS/clause references → reject
5. **Grounding verification** — ungrounded claims → reject
6. **Reflection agent** — final approval/rejection

**Rejection response** (from `KGRAGPipeline.UNKNOWN_RESPONSE`):
> "I cannot answer this question based on the available 3GPP documentation. The retrieved information is insufficient to provide a reliable, citation-backed response."

Note: The extractive generator may also return shorter rejection messages when excerpts are not relevant enough.

---

## 7. Data Flow (Example)

**Query:** "What is RRC connection establishment?"

1. Query router → intent: `DEFINITION` (`SemanticQueryRouter`)
2. Hybrid retriever → top spec chunks (e.g. from TS 38.300, 38.331, or other matching Rel-19 specs)
3. Require 3GPP evidence → pass (primary spec chunks found)
4. CoN → `can_answer: true` (via `MockLLM` + retrieval-confidence rules)
5. Extractive generator → verbatim excerpt with inline TS citation
6. Citation verifier → citations match retrieved chunks ✓
7. Grounding verifier → claims overlap retrieved text ✓
8. Reflection agent → `APPROVED` (via `MockLLM`)
9. Answer returned with confidence score and verification badges (Streamlit UI)

---

## 8. Technology Stack

| Layer | Technology |
|-------|------------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | NumPy cosine similarity |
| Keyword search | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Knowledge graph | NetworkX `DiGraph` + GraphML parsing (`xml.etree`) |
| Text splitting | langchain-text-splitters (hierarchical chunking) |
| UI | Streamlit (`app/streamlit_app.py`) + CLI (`scripts/chat_cli.py`) |
| Config | pydantic-settings |
| Tests | pytest (10 tests) |

---

## 9. Setup & Deployment

```bash
cd RAG
pip install -r requirements.txt
copy .env.example .env
python scripts/download_data.py   # ~300 MB from Hugging Face
python scripts/build_index.py     # ~5 min to embed 22K chunks
streamlit run app/streamlit_app.py
```

Dataset and index are **not included** in the repository (too large for Git). They are generated locally via the scripts above.

---

## 10. Known Limitations

- **Extractive wording** — answers use technical spec language, not simplified English
- **Dataset coverage** — limited to 3GPP Rel-19 content in the HF dataset
- **Text reconstruction** — 3GPP text rebuilt from KG descriptions, not raw PDFs
- **Indirect questions** — broad queries may be rejected if retrieval confidence is low
- **No external LLM** — verification agents are rule-based

---

## 11. Future Improvements

- Ingest raw 3GPP PDF/HTML for fuller text coverage
- Cross-encoder reranking for improved retrieval precision
- Evaluation benchmark with labeled 3GPP Q&A pairs
- Optional read-only LLM summarization with strict citation constraints

---

## 12. Repository

**GitHub:** https://github.com/pratistha9903/3GPP-RAG-Chatbot
