"""Structure-aware hierarchical chunking for 3GPP documents."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    """A single chunk with full provenance metadata."""

    chunk_id: str
    text: str
    spec_id: str = ""
    release: str = "Rel-19"
    clause: str = ""
    title_path: str = ""
    parent_id: str | None = None
    is_parent: bool = False
    source_file: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "spec_id": self.spec_id,
            "release": self.release,
            "clause": self.clause,
            "title_path": self.title_path,
            "parent_id": self.parent_id,
            "is_parent": self.is_parent,
            "source_file": self.source_file,
            **self.metadata,
        }


# Patterns for 3GPP document structure
CLAUSE_PATTERN = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)\s+(?P<title>.+)$", re.MULTILINE
)
SPEC_HEADER_PATTERN = re.compile(
    r"^(?:TS|TR)\s*(?P<spec>\d+\.\d+)", re.IGNORECASE | re.MULTILINE
)


class HierarchicalChunker:
    """
    Structure-aware chunker that preserves section/clause boundaries.

    Implements parent-child chunking:
    - Parent chunks: full section with title path context
    - Child chunks: smaller segments for precise retrieval
    """

    def __init__(
        self,
        child_chunk_size: int = 1250,
        child_chunk_overlap: int = 100,
    ):
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(
        self,
        text: str,
        spec_id: str = "",
        release: str = "Rel-19",
        source_file: str = "",
    ) -> list[DocumentChunk]:
        """Split a 3GPP document into hierarchical parent-child chunks."""
        if not spec_id:
            match = SPEC_HEADER_PATTERN.search(text[:500])
            spec_id = match.group("spec") if match else "unknown"

        sections = self._split_into_sections(text)
        all_chunks: list[DocumentChunk] = []

        for section in sections:
            parent_id = f"parent-{uuid.uuid4().hex[:12]}"
            title_path = section["title_path"]
            clause = section["clause"]
            section_text = section["text"]

            context_prefix = self._build_context_prefix(
                spec_id, release, title_path, clause
            )
            parent_text = f"{context_prefix}\n\n{section_text}".strip()

            parent_chunk = DocumentChunk(
                chunk_id=parent_id,
                text=parent_text,
                spec_id=spec_id,
                release=release,
                clause=clause,
                title_path=title_path,
                is_parent=True,
                source_file=source_file,
            )
            all_chunks.append(parent_chunk)

            child_texts = self.child_splitter.split_text(section_text)
            for i, child_text in enumerate(child_texts):
                if len(child_text.strip()) < 50:
                    continue
                child_with_context = f"{context_prefix}\n\n{child_text}".strip()
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=f"child-{uuid.uuid4().hex[:12]}",
                        text=child_with_context,
                        spec_id=spec_id,
                        release=release,
                        clause=clause,
                        title_path=title_path,
                        parent_id=parent_id,
                        is_parent=False,
                        source_file=source_file,
                    )
                )

        return all_chunks

    def _split_into_sections(self, text: str) -> list[dict]:
        """Split document by clause headings while preserving hierarchy."""
        lines = text.split("\n")
        sections: list[dict] = []
        current_clause = ""
        current_title = ""
        title_stack: list[tuple[str, str]] = []
        current_lines: list[str] = []

        for line in lines:
            match = CLAUSE_PATTERN.match(line.strip())
            if match and len(match.group("num").split(".")) <= 4:
                if current_lines:
                    sections.append(
                        self._make_section(
                            current_clause, current_title, title_stack, current_lines
                        )
                    )
                clause_num = match.group("num")
                title = match.group("title").strip()
                self._update_title_stack(title_stack, clause_num, title)
                current_clause = clause_num
                current_title = title
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                self._make_section(
                    current_clause, current_title, title_stack, current_lines
                )
            )

        if not sections:
            sections.append(
                {
                    "clause": "",
                    "title_path": "Document Root",
                    "text": text.strip(),
                }
            )

        return sections

    def _update_title_stack(
        self, stack: list[tuple[str, str]], clause: str, title: str
    ) -> None:
        depth = len(clause.split("."))
        while len(stack) >= depth:
            stack.pop()
        stack.append((clause, title))

    def _make_section(
        self,
        clause: str,
        title: str,
        stack: list[tuple[str, str]],
        lines: list[str],
    ) -> dict:
        title_path = " > ".join(f"{c} {t}" for c, t in stack) if stack else title
        return {
            "clause": clause,
            "title_path": title_path or "Document Root",
            "text": "\n".join(lines).strip(),
        }

    @staticmethod
    def _build_context_prefix(
        spec_id: str, release: str, title_path: str, clause: str
    ) -> str:
        parts = [f"[{release} TS {spec_id}]"]
        if clause:
            parts.append(f"Section {clause}")
        if title_path:
            parts.append(title_path)
        return " | ".join(parts)

    def chunk_from_kg_node(
        self,
        node_id: str,
        description: str,
        entity_type: str,
        chunk_id: str,
        source_file: str,
        release: str = "Rel-19",
    ) -> DocumentChunk | None:
        """Create a chunk from a knowledge graph node description."""
        if not description or len(description.strip()) < 20:
            return None

        spec_id = self._extract_spec_from_path(source_file)
        text = f"[KG Entity: {node_id}] ({entity_type})\n{description.strip()}"

        return DocumentChunk(
            chunk_id=f"kg-{chunk_id or uuid.uuid4().hex[:12]}",
            text=text,
            spec_id=spec_id,
            release=release,
            source_file=source_file,
            metadata={"entity_type": entity_type, "kg_node_id": node_id},
        )

    @staticmethod
    def _extract_spec_from_path(path: str) -> str:
        """Extract TS number from a 3GPP file path like 21900-j00.md3."""
        match = re.search(r"/(\d{5})-", path.replace("\\", "/"))
        if match:
            num = match.group(1)
            return f"{num[:2]}.{num[2:]}"
        match = re.search(r"(\d{2})(\d{3})", path)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        return "unknown"
