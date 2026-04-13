"""
RAG (Retrieval-Augmented Generation) module for lesson-aware AI tutoring.

Responsibilities:
  1.  Receive lesson materials (text / extracted-PDF content).
  2.  Chunk the materials into manageable pieces.
  3.  Embed and index chunks in an ephemeral ChromaDB collection.
  4.  At query time, retrieve the most relevant chunks for the student's
      question or the current point in the lesson.
"""

# ChromaDB needs sqlite3 >= 3.35.  On older systems we swap in pysqlite3.
try:
    import pysqlite3
    import sys

    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import json
import logging
import uuid
from typing import Optional

import chromadb
from openai import OpenAI

logger = logging.getLogger(__name__)

# ─── Chunking parameters ───
CHUNK_SIZE = 600  # characters per chunk
CHUNK_OVERLAP = 100  # overlap between consecutive chunks


def _chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split *text* into overlapping windows of roughly *chunk_size* chars."""
    if not text or not text.strip():
        return []

    # Normalise whitespace
    text = " ".join(text.split())

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


class LessonRAG:
    """Per-session RAG context for a single lesson.

    Lifecycle:
      1. ``__init__`` — pass the shared OpenAI client.
      2. ``ingest(lesson_context_json)`` — parse & index the materials.
      3. ``retrieve(query, k)``          — fetch top-k relevant chunks.
      4. ``build_system_prompt()``       — return a rich system prompt.
    """

    def __init__(self, openai_client: OpenAI):
        self._openai = openai_client
        self._collection: Optional[chromadb.Collection] = None
        self._client = chromadb.Client()  # in-memory, ephemeral

        # Metadata populated by ingest()
        self.lesson_title: str = ""
        self.course_title: str = ""
        self.class_title: str = ""
        self.material_count: int = 0
        self.chunk_count: int = 0
        self._all_chunks: list[str] = []
        self._lesson_outline: str = ""  # short summary for prompt

    # ──────────────────────────────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────────────────────────────

    def ingest(self, lesson_context_json: str) -> None:
        """Parse the JSON context sent by the backend and build the index."""
        try:
            ctx = json.loads(lesson_context_json)
        except json.JSONDecodeError:
            logger.error("Failed to decode lesson_context JSON")
            return

        self.lesson_title = ctx.get("lesson_title", "Unknown lesson")
        self.course_title = ctx.get("course_title", "Unknown course")
        self.class_title = ctx.get("class_title", "")
        materials: list[dict] = ctx.get("materials", [])

        logger.info(
            "RAG ingest — lesson=%r  course=%r  materials=%d",
            self.lesson_title,
            self.course_title,
            len(materials),
        )

        # Chunk all materials
        all_chunks: list[str] = []
        chunk_metas: list[dict] = []

        for mat in materials:
            mat_type = mat.get("type", "text")
            content = mat.get("content", "")
            title = mat.get("title", "")

            if not content or not content.strip():
                continue

            chunks = _chunk_text(content)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metas.append(
                    {
                        "material_type": mat_type,
                        "material_title": title,
                        "chunk_index": str(i),
                    }
                )

        if not all_chunks:
            logger.warning("RAG: no chunks produced — lesson has no usable content")
            return

        self._all_chunks = all_chunks
        self.material_count = len(materials)
        self.chunk_count = len(all_chunks)

        # ── Embed with OpenAI ──
        logger.info("RAG: embedding %d chunks with OpenAI …", self.chunk_count)
        embeddings = self._embed_texts(all_chunks)

        # ── Store in ChromaDB ──
        col_name = f"lesson_{uuid.uuid4().hex[:12]}"
        self._collection = self._client.create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection.add(
            ids=[f"chunk_{i}" for i in range(len(all_chunks))],
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=chunk_metas,
        )

        # Build a short outline of the content for the system prompt
        self._lesson_outline = self._build_outline(all_chunks)
        logger.info(
            "RAG: indexed %d chunks into collection %r", self.chunk_count, col_name
        )

    # ──────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        """Return the *k* most relevant document chunks for *query*."""
        if self._collection is None or self.chunk_count == 0:
            return []

        q_emb = self._embed_texts([query])
        results = self._collection.query(
            query_embeddings=q_emb,
            n_results=min(k, self.chunk_count),
        )
        docs = results.get("documents", [[]])[0]
        logger.info("RAG retrieve: query=%r → %d chunks", query[:60], len(docs))
        return docs

    # ──────────────────────────────────────────────────────────────────
    # System prompt construction
    # ──────────────────────────────────────────────────────────────────

    def build_system_prompt(self) -> str:
        """Return a system prompt that makes the AI actively teach the lesson."""

        base_prompt = (
            "You are an expert AI tutor conducting a live voice lesson with a student. "
            "Your role is to ACTIVELY TEACH and DRIVE the lesson — do not passively wait for questions.\n\n"
            "## Your Teaching Approach\n"
            "1. **Start by introducing the topic**: greet the student, tell them what today's lesson is about, "
            "and give a brief overview of what you'll cover.\n"
            "2. **Teach progressively**: explain concepts one at a time, building from simple to complex.\n"
            "3. **Check understanding**: after explaining a concept, ask the student a question to verify they understood.\n"
            "4. **Encourage participation**: ask open-ended questions, request examples, and invite the student to explain back.\n"
            "5. **Provide feedback**: praise correct answers, gently correct mistakes, and offer additional explanation when needed.\n"
            "6. **Summarise periodically**: recap what has been covered before moving to the next topic.\n"
            "7. **Adapt**: if the student seems confused, slow down and re-explain. If they're ahead, move faster.\n\n"
            "## Voice Conversation Rules\n"
            "- Keep responses concise and natural for speech (2–4 sentences typically).\n"
            "- Avoid markdown formatting, bullet points, code blocks, or anything that cannot be read aloud.\n"
            "- Use a warm, encouraging, conversational tone.\n"
            "- Do NOT dump all information at once — teach interactively.\n"
            "- When you ask a question, wait for the student's response before continuing.\n\n"
        )

        if self.lesson_title:
            base_prompt += f"## Lesson Details\n" f"- **Course**: {self.course_title}\n"
            if self.class_title:
                base_prompt += f"- **Unit/Class**: {self.class_title}\n"
            base_prompt += f"- **Lesson**: {self.lesson_title}\n\n"

        if self._lesson_outline:
            base_prompt += (
                "## Lesson Content Overview\n"
                "Below is an outline of the material you should teach. "
                "Use this to structure your lesson, covering topics in a logical order:\n\n"
                f"{self._lesson_outline}\n\n"
            )

        base_prompt += (
            "## Important\n"
            "You have access to the full lesson materials through a retrieval system. "
            "When discussing specific topics, you will be provided with the relevant excerpts. "
            "Always base your teaching on the actual lesson content — do not invent facts. "
            "If the student asks about something outside the lesson scope, briefly acknowledge it "
            "and guide them back to the current lesson material.\n\n"
            "Begin the lesson now by greeting the student and introducing today's topic."
        )

        return base_prompt

    def build_retrieval_context(self, user_message: str) -> str:
        """Given a student message, retrieve relevant chunks and format them."""
        chunks = self.retrieve(user_message, k=4)
        if not chunks:
            return ""

        context_parts = [
            "[Relevant lesson material for your reference — use this to inform your response]"
        ]
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"--- Excerpt {i} ---\n{chunk}")

        return "\n\n".join(context_parts)

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Get OpenAI embeddings for a list of texts."""
        response = self._openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

    @staticmethod
    def _build_outline(chunks: list[str], max_chars: int = 1500) -> str:
        """Create a condensed outline from the first chunks of material."""
        outline_parts = []
        total = 0
        for chunk in chunks:
            if total + len(chunk) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    outline_parts.append(chunk[:remaining] + "…")
                break
            outline_parts.append(chunk)
            total += len(chunk)
        return "\n".join(outline_parts)

    def close(self) -> None:
        """Clean up the in-memory collection."""
        if self._collection is not None:
            try:
                self._client.delete_collection(self._collection.name)
            except Exception:
                pass
            self._collection = None
