"""
test_rag.py — Unit tests for the RAG pipeline.

Tests chunking, ingestion, embedding, retrieval, and prompt generation
WITHOUT requiring the full audio/gRPC/TTS stack.

Run:
    pip install chromadb openai python-dotenv pytest
    OPENAI_API_KEY=sk-... pytest __tests__/test_rag.py -v
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

# ─── Stub old openai if OpenAI class is missing ───
try:
    from openai import OpenAI  # noqa: F401
except (ImportError, AttributeError):
    import openai as _openai_mod

    _openai_mod.OpenAI = MagicMock()

# ─── Ensure env is loaded ───
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────
# Unit tests for _chunk_text (no external dependencies)
# ─────────────────────────────────────────────────────────────────────

from core.rag import _chunk_text, LessonRAG


class _FakeCollection:
    def __init__(self, name: str):
        self.name = name
        self._documents: list[str] = []

    def add(self, ids, embeddings, documents, metadatas):
        self._documents.extend(documents)

    def query(self, query_embeddings, n_results):
        return {"documents": [self._documents[:n_results]]}


class _FakeChromaClient:
    def __init__(self):
        self._collections = {}

    def create_collection(self, name, metadata=None):
        collection = _FakeCollection(name)
        self._collections[name] = collection
        return collection

    def delete_collection(self, name):
        self._collections.pop(name, None)


@pytest.fixture(autouse=True)
def _patch_chromadb_client():
    with patch("core.rag.chromadb.Client", return_value=_FakeChromaClient()):
        yield


class TestChunking:
    def test_empty_text(self):
        assert _chunk_text("") == []
        assert _chunk_text("   ") == []

    def test_short_text(self):
        chunks = _chunk_text("Hello world", chunk_size=600, overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_overlap(self):
        # 20 chars, chunk=10, overlap=5 → windows at [0:10], [5:15], [10:20]
        text = "abcdefghij" * 2  # 20 chars
        chunks = _chunk_text(text, chunk_size=10, overlap=5)
        assert len(chunks) >= 2
        # Verify overlap: end of chunk[0] overlaps with start of chunk[1]
        assert chunks[0][-5:] == chunks[1][:5]

    def test_real_content(self):
        content = (
            "Photosynthesis is the process by which green plants and some other organisms "
            "use sunlight to synthesize foods from carbon dioxide and water. Photosynthesis "
            "in plants generally involves the green pigment chlorophyll and generates oxygen "
            "as a byproduct. The process occurs mainly in the leaves, specifically in the "
            "chloroplasts. There are two stages of photosynthesis: the light-dependent reactions "
            "and the Calvin cycle. The light-dependent reactions take place in the thylakoid "
            "membranes and require sunlight. The Calvin cycle occurs in the stroma of the "
            "chloroplast and does not require light directly."
        )
        chunks = _chunk_text(content, chunk_size=200, overlap=50)
        assert len(chunks) >= 2
        # All original content should be covered
        joined = " ".join(chunks)
        assert "Photosynthesis" in joined
        assert "Calvin cycle" in joined


# ─────────────────────────────────────────────────────────────────────
# Unit tests for LessonRAG with mocked embeddings
# ─────────────────────────────────────────────────────────────────────


def _make_fake_embedding(dim: int = 1536):
    """Return a deterministic but varying embedding."""
    import hashlib

    def embed(texts):
        results = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [float(b) / 255.0 for b in h]
            # Pad / truncate to dim
            vec = (vec * (dim // len(vec) + 1))[:dim]
            results.append(vec)
        return results

    return embed


class TestLessonRAGWithMock:
    """Tests using mocked OpenAI embeddings to avoid API calls."""

    @pytest.fixture()
    def rag(self):
        client = MagicMock()

        # Mock embeddings.create to return deterministic vectors
        fake_embed = _make_fake_embedding()

        def mock_create(model, input):
            vecs = fake_embed(input)
            items = [MagicMock(embedding=v) for v in vecs]
            result = MagicMock()
            result.data = items
            return result

        client.embeddings.create = mock_create
        return LessonRAG(client)

    def test_ingest_text_material(self, rag):
        ctx = json.dumps(
            {
                "lesson_title": "Introduction to Photosynthesis",
                "course_title": "Biology 101",
                "class_title": "Unit 3: Plant Biology",
                "materials": [
                    {
                        "type": "text",
                        "content": (
                            "Photosynthesis is the process by which green plants use "
                            "sunlight to synthesize foods from carbon dioxide and water. "
                            "The process occurs mainly in the leaves through chloroplasts. "
                            "There are two main stages: light-dependent reactions in the "
                            "thylakoid membranes, and the Calvin cycle in the stroma."
                        ),
                        "title": "Lesson Notes",
                    }
                ],
            }
        )
        rag.ingest(ctx)

        assert rag.lesson_title == "Introduction to Photosynthesis"
        assert rag.course_title == "Biology 101"
        assert rag.class_title == "Unit 3: Plant Biology"
        assert rag.chunk_count > 0
        assert rag.material_count == 1

    def test_ingest_multiple_materials(self, rag):
        ctx = json.dumps(
            {
                "lesson_title": "Cell Division",
                "course_title": "Biology 101",
                "class_title": "",
                "materials": [
                    {
                        "type": "text",
                        "content": "Mitosis is a type of cell division.",
                        "title": "Notes",
                    },
                    {
                        "type": "text",
                        "content": "Meiosis produces gametes with half the chromosomes.",
                        "title": "Extra",
                    },
                ],
            }
        )
        rag.ingest(ctx)
        assert rag.material_count == 2
        assert rag.chunk_count >= 2

    def test_ingest_empty_materials(self, rag):
        ctx = json.dumps(
            {
                "lesson_title": "Empty Lesson",
                "course_title": "Test",
                "materials": [],
            }
        )
        rag.ingest(ctx)
        assert rag.chunk_count == 0

    def test_retrieval_returns_chunks(self, rag):
        ctx = json.dumps(
            {
                "lesson_title": "Photosynthesis",
                "course_title": "Biology",
                "materials": [
                    {
                        "type": "text",
                        "content": (
                            "Photosynthesis converts light energy into chemical energy. "
                            "It takes place in the chloroplasts of plant cells. "
                            "The Calvin cycle is the second stage of photosynthesis. "
                            "It uses carbon dioxide and ATP to produce glucose. "
                            "Chlorophyll is the green pigment that absorbs sunlight."
                        ),
                        "title": "Notes",
                    }
                ],
            }
        )
        rag.ingest(ctx)
        results = rag.retrieve("What is the Calvin cycle?", k=2)
        assert len(results) > 0
        assert any(
            "Calvin" in r or "cycle" in r or "photosynthesis" in r.lower()
            for r in results
        )

    def test_build_system_prompt(self, rag):
        ctx = json.dumps(
            {
                "lesson_title": "The Water Cycle",
                "course_title": "Earth Science",
                "class_title": "Unit 2",
                "materials": [
                    {
                        "type": "text",
                        "content": "Evaporation, condensation, precipitation.",
                        "title": "Notes",
                    }
                ],
            }
        )
        rag.ingest(ctx)
        prompt = rag.build_system_prompt()

        # The prompt should mention the lesson
        assert "Water Cycle" in prompt
        assert "Earth Science" in prompt
        # The prompt should instruct lesson-driving behavior
        assert "TEACH" in prompt or "teach" in prompt
        assert "greeting" in prompt.lower() or "introduce" in prompt.lower()

    def test_build_retrieval_context(self, rag):
        ctx = json.dumps(
            {
                "lesson_title": "Gravity",
                "course_title": "Physics",
                "materials": [
                    {
                        "type": "text",
                        "content": "Gravity is a fundamental force of nature that attracts objects with mass.",
                        "title": "Notes",
                    }
                ],
            }
        )
        rag.ingest(ctx)
        context = rag.build_retrieval_context("Tell me about gravity")
        assert "Relevant lesson material" in context
        assert "Excerpt" in context

    def test_retrieval_empty_when_no_materials(self, rag):
        ctx = json.dumps(
            {
                "lesson_title": "Empty",
                "course_title": "Test",
                "materials": [],
            }
        )
        rag.ingest(ctx)
        results = rag.retrieve("anything")
        assert results == []
        context = rag.build_retrieval_context("anything")
        assert context == ""

    def test_close(self, rag):
        ctx = json.dumps(
            {
                "lesson_title": "Test",
                "course_title": "Test",
                "materials": [
                    {"type": "text", "content": "Some content here.", "title": "N"}
                ],
            }
        )
        rag.ingest(ctx)
        assert rag._collection is not None
        rag.close()
        assert rag._collection is None


# ─────────────────────────────────────────────────────────────────────
# Integration test with REAL OpenAI API (skipped if no key)
# ─────────────────────────────────────────────────────────────────────

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")


@pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
class TestLessonRAGIntegration:
    """Integration tests that call the real OpenAI embedding API."""

    @pytest.fixture()
    def rag(self):
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_KEY)
        r = LessonRAG(client)
        yield r
        r.close()

    def test_full_rag_cycle(self, rag):
        """Ingest real content, embed with OpenAI, retrieve relevant chunks."""
        lesson_ctx = json.dumps(
            {
                "lesson_title": "Newton's Laws of Motion",
                "course_title": "Physics 101",
                "class_title": "Classical Mechanics",
                "materials": [
                    {
                        "type": "text",
                        "content": (
                            "Newton's First Law states that an object at rest stays at rest "
                            "and an object in motion stays in motion unless acted upon by an "
                            "external force. This is also known as the law of inertia. "
                            "Newton's Second Law states that force equals mass times acceleration, "
                            "expressed as F = ma. This law explains how the velocity of an object "
                            "changes when it is subjected to an external force. "
                            "Newton's Third Law states that for every action, there is an equal "
                            "and opposite reaction. When one body exerts a force on a second body, "
                            "the second body simultaneously exerts a force equal in magnitude and "
                            "opposite in direction on the first body."
                        ),
                        "title": "Lecture Notes",
                    },
                    {
                        "type": "text",
                        "content": (
                            "Example problems: A 5 kg object is pushed with a force of 20 N. "
                            "What is its acceleration? Using F = ma, a = F/m = 20/5 = 4 m/s². "
                            "A car of mass 1000 kg accelerates at 3 m/s². What is the net force? "
                            "F = ma = 1000 × 3 = 3000 N."
                        ),
                        "title": "Practice Problems",
                    },
                ],
            }
        )

        rag.ingest(lesson_ctx)

        assert rag.lesson_title == "Newton's Laws of Motion"
        assert rag.chunk_count > 0

        # Retrieve chunks about F = ma
        results = rag.retrieve("What is Newton's second law?", k=3)
        assert len(results) > 0
        # At least one chunk should mention F = ma or second law
        combined = " ".join(results).lower()
        assert "f = ma" in combined or "second law" in combined or "force" in combined

        # The system prompt should be lesson-aware
        prompt = rag.build_system_prompt()
        assert "Newton" in prompt
        assert "Physics 101" in prompt

        # Build retrieval context
        ctx = rag.build_retrieval_context("Explain F equals ma")
        assert "Excerpt" in ctx
        assert len(ctx) > 50

        print("\n✅ Full RAG integration test passed!")
        print(f"   Lesson: {rag.lesson_title}")
        print(f"   Chunks indexed: {rag.chunk_count}")
        print(f"   Retrieved chunks for 'F=ma' query: {len(results)}")
        print(f"   System prompt length: {len(prompt)} chars")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
