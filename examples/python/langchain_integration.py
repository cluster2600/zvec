#!/usr/bin/env python3
# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LangChain + zvec integration example.

Prerequisites
-------------
Install the required packages::

    pip install zvec langchain-core

For real embeddings (optional, used in the second example)::

    pip install langchain-openai

Usage
-----
Run this script directly::

    python examples/python/langchain_integration.py
"""
from __future__ import annotations

import shutil
import tempfile

# ---------------------------------------------------------------------------
# 1. Basic example using LangChain's built-in FakeEmbeddings
# ---------------------------------------------------------------------------


def example_fake_embeddings() -> None:
    """Demonstrate ZvecVectorStore with FakeEmbeddings (no API key required)."""

    from langchain_core.embeddings import FakeEmbeddings

    from zvec.langchain.vectorstore import ZvecVectorStore

    # Use a temporary directory that is cleaned up automatically.
    store_path = tempfile.mkdtemp(prefix="zvec_langchain_")
    try:
        print("=" * 60)
        print("Example 1 — FakeEmbeddings (no API key required)")
        print("=" * 60)

        # Documents to index.
        texts = [
            "zvec is an embedded vector database written in C++.",
            "LangChain is a framework for building LLM applications.",
            "Vector search enables semantic similarity retrieval.",
            "Python is a versatile programming language.",
            "Retrieval-Augmented Generation improves LLM accuracy.",
        ]
        metadatas = [
            {"source": "zvec-docs", "topic": "database"},
            {"source": "langchain-docs", "topic": "llm"},
            {"source": "ml-blog", "topic": "search"},
            {"source": "python-docs", "topic": "language"},
            {"source": "rag-paper", "topic": "llm"},
        ]

        # 384-dimensional fake embeddings (mimics sentence-transformers output size).
        embeddings = FakeEmbeddings(size=384)

        # Build the store from texts (creates collection, embeds, inserts).
        store = ZvecVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            path=store_path,
            dimension=384,
        )
        print(f"Indexed {len(texts)} documents into zvec at: {store_path}")

        # --- similarity_search ----------------------------------------
        query = "semantic similarity database"
        results = store.similarity_search(query, k=3)
        print(f"\nTop-3 results for query: '{query}'")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. [{doc.metadata.get('topic')}] {doc.page_content[:60]}")

        # --- similarity_search_with_score -----------------------------
        print(f"\nTop-2 results with scores for query: '{query}'")
        scored = store.similarity_search_with_score(query, k=2)
        for doc, score in scored:
            print(f"  score={score:.4f}  {doc.page_content[:60]}")

        # --- add_texts (incremental insert) ---------------------------
        new_texts = ["Embeddings represent text as dense float vectors."]
        new_ids = store.add_texts(new_texts, metadatas=[{"source": "tutorial"}])
        print(f"\nAdded 1 new document, ID: {new_ids[0]}")

        # --- Use as a LangChain retriever -----------------------------
        retriever = store.as_retriever(search_kwargs={"k": 2})
        retrieved = retriever.invoke("language model framework")
        print(f"\nRetriever returned {len(retrieved)} document(s):")
        for doc in retrieved:
            print(f"  • {doc.page_content[:60]}")

    finally:
        shutil.rmtree(store_path, ignore_errors=True)

    print("\nExample 1 complete.\n")


# ---------------------------------------------------------------------------
# 2. Example using OpenAI embeddings (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------


def example_openai_embeddings() -> None:
    """Demonstrate ZvecVectorStore with real OpenAI embeddings.

    Requires the ``langchain-openai`` package and an ``OPENAI_API_KEY``
    environment variable.
    """
    import os

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        print("Skipping OpenAI example — langchain-openai not installed.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping OpenAI example — OPENAI_API_KEY not set.")
        return

    from zvec.langchain.vectorstore import ZvecVectorStore

    store_path = tempfile.mkdtemp(prefix="zvec_openai_")
    try:
        print("=" * 60)
        print("Example 2 — OpenAI text-embedding-3-small")
        print("=" * 60)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        store = ZvecVectorStore.from_texts(
            texts=[
                "zvec stores vectors on disk for fast ANN search.",
                "OpenAI provides state-of-the-art text embeddings.",
                "LangChain chains together LLM calls and retrievers.",
            ],
            embedding=embeddings,
            path=store_path,
            # dimension is inferred automatically when not provided
        )

        results = store.similarity_search("disk-based vector index", k=2)
        print("Results:")
        for doc in results:
            print(f"  • {doc.page_content}")
    finally:
        shutil.rmtree(store_path, ignore_errors=True)

    print("\nExample 2 complete.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_fake_embeddings()
    example_openai_embeddings()
