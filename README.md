# 3GPP KG-RAG Chatbot

A **Knowledge Graph-enhanced Retrieval-Augmented Generation (KG-RAG)** chatbot for 3GPP telecom standards, designed to achieve **near-zero hallucinations** through extractive generation and multi-layer verification.

**Primary knowledge source:** [GSMA/telecom-kg-rel19](https://huggingface.co/datasets/GSMA/telecom-kg-rel19) on Hugging Face (3GPP Release 19 mirror).

No external LLM API keys required.



---

## Project Highlights

| Criterion | Implementation |
|-----------|----------------|
| RAG architecture | Hybrid retrieval (BM25 + semantic vectors + RRF) + KG triples |
| Near-zero hallucinations | Extractive-only answers, citation + grounding checks, reject when unsure |
| 3GPP primary source | `GSMA/telecom-kg-rel19` — 8,032 HF spec chunks + 14,072 KG supplement |
| Runnable demo | Streamlit UI, CLI, 10 automated tests |

---

## Zero-Hallucination Design

The system **never generates free-form text**. It only returns **verbatim quotes** from retrieved 3GPP documentation, then verifies them before release.

| Guardrail | Behavior |
|-----------|----------|
| **Primary 3GPP evidence required** | Rejects if no HF spec chunks are retrieved |
| **Extractive generation** | Verbatim quotes only — no external LLM |
| **Chain-of-Noting (CoN)** | Rejects when retrieved context is insufficient |
| **Citation verification** | Rejects citations not present in retrieved chunks |
| **Grounding verification** | Rejects claims not supported by retrieved text |
| **Reflection agent** | Final rule-based cross-validation |

When evidence is insufficient, the chatbot responds:

> *"I cannot answer this question based on the available 3GPP documentation."*

---

## Primary Knowledge Source

> Our primary knowledge source is official **3GPP Release 19** standards documentation, accessed through the [GSMA/telecom-kg-rel19](https://huggingface.co/datasets/GSMA/telecom-kg-rel19) dataset on Hugging Face (GSMA Open Telco Assets Initiative).

| Component | Source | Role |
|-----------|--------|------|
| **Primary corpus** | `GSMA/telecom-kg-rel19` | 3GPP spec text (authoritative) |
| **3GPP text** | KG-grounded chunk descriptions | ~8,032 primary chunks |
| **Chunk metadata** | `rel19_text_chunks.jsonl` | 896K chunk IDs with TS/release provenance |
| **KG supplement** | `rel19_3gpp_telecom_kg.graphml` | 21K+ nodes, 31K+ edges for relationship queries |

**Built index:** 22,104 total chunks (HF primary + KG supplement), 384-dim embeddings.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│  Query Router            │  ← Intent classification + query expansion
└────────────┬────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────┐   ┌──────────────┐
│ Hybrid   │   │ KG Triple    │
│ Retriever│   │ Retriever    │
│ BM25 +   │   │ (Ontology)   │
│ Semantic │   │              │
│ + RRF    │   │              │
└────┬─────┘   └──────┬───────┘
     │                │
     └────────┬───────┘
              ▼
┌─────────────────────────┐
│  Chain-of-Noting (CoN)   │  ← Rejects if context insufficient
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Extractive Generator    │  ← Verbatim 3GPP quotes only
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Citation Verifier       │  ← Flags hallucinated references
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Grounding Verifier      │  ← Checks claims vs retrieved text
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Reflection Agent        │  ← Final cross-validation
└────────────┬────────────┘
             ▼
         Final Answer
```

---

## Key Features

### Retrieval & Indexing
- **Hybrid retrieval** — BM25 keyword search + dense vector semantic search, fused with Reciprocal Rank Fusion (RRF)
- **Query expansion** — maps everyday terms to 3GPP vocabulary (e.g. "phone" → UE, "connect" → RRC/attach/register)
- **Structure-aware chunking** — preserves section/clause boundaries from 3GPP specs
- **Parent-child enrichment** — adds parent section context to child chunks
- **Source priority** — primary HF 3GPP chunks boosted 3× over KG supplement

### Knowledge Graph (KG-RAG)
- Loads `rel19_3gpp_telecom_kg.graphml` from the HF dataset
- Retrieves ontology triples for relationship queries (e.g. entity interactions)
- Used as **supplement only** — primary answers must come from 3GPP spec text

### Generation & Verification
- **Extractive answers** — selects the most relevant verbatim excerpts from retrieved chunks
- **Mandatory citations** — every answer includes TS number, release, and clause where available
- **Multi-layer rejection** — off-topic or ungrounded queries are rejected, not guessed

---

## Quick Start

### Prerequisites
- Python 3.10+
- ~2 GB disk space (dataset + index)

### 1. Install

```bash
cd RAG
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/Mac
```

Default settings enable strict zero-hallucination mode. **No API keys needed.**

### 3. Download data & build index

```bash
python scripts/download_data.py   # Download from Hugging Face (~300 MB)
python scripts/build_index.py     # Build hybrid search index (~5 min)
```

Expected output:
```
HF 3GPP (primary):  8032
KG (supplement):    14072
Total:              22104
```

### 4. Run the chatbot

**Streamlit Web UI:**
```bash
streamlit run app/streamlit_app.py
```

**CLI:**
```bash
python scripts/chat_cli.py
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Sample Questions

**Likely to answer** (if covered in Rel-19 dataset):
- "What is RRC connection establishment?"
- "What are the functions of the AMF?"
- "What is the purpose of SIB1?"
- "How does network slicing work in 5G?"
- "What is network selection in 3GPP?"

**Expected to reject** (not in dataset / ungrounded):
- "What is blockchain?"
- "Who won the World Cup?"
- Topics outside 3GPP Rel-19 coverage

---

## Project Structure

```
RAG/
├── app/                    # Streamlit web UI
├── config/                 # Settings (.env)
├── data/
│   ├── tkg/                # Knowledge graph (downloaded)
│   ├── chunks/             # HF chunk metadata (downloaded)
│   ├── mappings/           # Entity mappings (downloaded)
│   └── index/              # Built search index (embeddings + BM25)
├── scripts/
│   ├── download_data.py    # Download GSMA/telecom-kg-rel19
│   ├── build_index.py      # Build hybrid search index
│   └── chat_cli.py         # Command-line interface
├── src/
│   ├── indexing/           # Chunking, HF loader, embedding, indexing
│   ├── retrieval/          # Hybrid retriever, query router, KG retriever
│   ├── knowledge_graph/    # GraphML loader
│   ├── generation/         # Extractive generator, verification agents
│   ├── verification/       # CoN, citation verifier, grounding verifier
│   └── pipeline/           # End-to-end orchestration
└── tests/                  # Automated test suite
```

---

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `STRICT_ZERO_HALLUCINATION` | `true` | Enable all verification guardrails |
| `REQUIRE_3GPP_SPEC_EVIDENCE` | `true` | Require primary HF spec chunks |
| `REJECT_ON_INVALID_CITATION` | `true` | Reject hallucinated TS/clause references |
| `REJECT_ON_UNGROUNDED_CLAIM` | `true` | Reject claims not in retrieved text |
| `MIN_CON_CONFIDENCE` | `0.75` | Minimum Chain-of-Noting confidence |
| `MIN_GROUNDING_SCORE` | `0.45` | Minimum grounding overlap score |
| `INDEX_KG_SUPPLEMENT` | `true` | Include KG chunks in index |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search | NumPy cosine similarity |
| Keyword search | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Knowledge graph | NetworkX + GraphML |
| Verification | Rule-based CoN + reflection agents |
| UI | Streamlit |
| Config | pydantic-settings |

---

## Known Limitations

- **Extractive wording** — answers quote spec text directly (technical language, not simplified English)
- **Dataset coverage** — limited to 3GPP Rel-19 content present in `GSMA/telecom-kg-rel19`
- **HF text chunks** — `rel19_text_chunks.jsonl` contains metadata only; 3GPP text is reconstructed from KG-grounded descriptions
- **Indirect questions** — broad questions may be rejected if retrieval confidence is below threshold

---

## License

The [GSMA/telecom-kg-rel19](https://huggingface.co/datasets/GSMA/telecom-kg-rel19) dataset is released under **CC BY-NC 4.0**.


