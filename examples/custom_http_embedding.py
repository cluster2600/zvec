"""
Custom HTTP Embedding Example for zvec
======================================

Demonstrates how to use any OpenAI-compatible embedding endpoint
(LM Studio, Ollama, vLLM, LocalAI, …) as an embedding source in zvec.

Usage
-----
1. Start your local inference server:

   **LM Studio** (https://lmstudio.ai/):
       - Open LM Studio → go to "Local Server" tab
       - Load an embedding model (e.g. nomic-embed-text, all-minilm-l6-v2)
       - Click "Start Server" (default: http://localhost:1234)
       - Enable "Allow external connections" if accessing from another machine

   **Ollama** (https://ollama.com/):
       $ ollama serve                      # starts on http://localhost:11434
       $ ollama pull nomic-embed-text      # pull the model first

2. Install zvec:
       $ pip install zvec

3. Run the example:
       # LM Studio (default)
       $ python examples/custom_http_embedding.py

       # Ollama
       $ python examples/custom_http_embedding.py \\
             --base-url http://localhost:11434 \\
             --model nomic-embed-text

       # Remote / custom server
       $ python examples/custom_http_embedding.py \\
             --base-url http://192.168.1.10:1234 \\
             --model text-embedding-nomic-embed-text-v1.5@f16

Notes
-----
- The embedding dimension is detected automatically on the first call.
- No API key is required for local servers; pass ``--api-key`` if yours needs one.
- The collection is stored under ``/tmp/zvec_http_embedding_example`` and is
  destroyed at the end of the script.  Remove the ``collection.destroy()`` call
  at the bottom to keep the data across runs.
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Optional

from zvec.extension import HTTPDenseEmbedding


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_1",
        "text": "LM Studio lets you run large language models locally on your computer.",
        "topic": "local AI",
    },
    {
        "id": "doc_2",
        "text": "Ollama is an open-source tool for running language models on-device.",
        "topic": "local AI",
    },
    {
        "id": "doc_3",
        "text": "zvec is a lightweight, in-process vector database built on Proxima.",
        "topic": "vector database",
    },
    {
        "id": "doc_4",
        "text": "HNSW is a graph-based algorithm for approximate nearest-neighbor search.",
        "topic": "ANN algorithms",
    },
    {
        "id": "doc_5",
        "text": "Cosine similarity measures the angle between two vectors, ignoring magnitude.",
        "topic": "math",
    },
]

QUERY = "How do I run an embedding model on my laptop?"


def run_demo(
    base_url: str,
    model: str,
    api_key: Optional[str],
    collection_path: str,
) -> None:
    import zvec
    from zvec import (
        CollectionSchema,
        DataType,
        Doc,
        HnswIndexParam,
        MetricType,
        VectorQuery,
        VectorSchema,
        create_and_open,
    )

    # ------------------------------------------------------------------ #
    # 1.  Embedding function                                               #
    # ------------------------------------------------------------------ #
    print(f"[1/4] Connecting to embedding server at {base_url} …")
    emb = HTTPDenseEmbedding(base_url=base_url, model=model, api_key=api_key)

    # Probe dimension
    dim = emb.dimension
    print(f"      Model: {model!r}  |  Dimension: {dim}")

    # ------------------------------------------------------------------ #
    # 2.  Create collection with HNSW + cosine                            #
    # ------------------------------------------------------------------ #
    print("[2/4] Creating zvec collection (HNSW / cosine) …")
    if os.path.exists(collection_path):
        shutil.rmtree(collection_path)

    schema = CollectionSchema(
        name="http_embedding_demo",
        vectors=VectorSchema(
            name="embedding",
            data_type=DataType.VECTOR_FP32,
            dimension=dim,
            index_param=HnswIndexParam(
                metric_type=MetricType.COSINE,
                m=16,
                ef_construction=200,
            ),
        ),
    )
    collection = create_and_open(path=collection_path, schema=schema)

    # ------------------------------------------------------------------ #
    # 3.  Insert documents                                                 #
    # ------------------------------------------------------------------ #
    print(f"[3/4] Embedding and inserting {len(SAMPLE_DOCUMENTS)} documents …")
    docs = []
    for item in SAMPLE_DOCUMENTS:
        vector = emb.embed(item["text"])
        doc = Doc(
            id=item["id"],
            vectors={"embedding": vector},
            fields={
                "text": item["text"],
                "topic": item["topic"],
            },
        )
        docs.append(doc)

    collection.insert(docs)
    collection.flush()
    print(f"      Inserted {collection.stats.total_doc_count} documents.")

    # ------------------------------------------------------------------ #
    # 4.  Search                                                           #
    # ------------------------------------------------------------------ #
    print(f"[4/4] Searching for: {QUERY!r}\n")
    query_vector = emb.embed(QUERY)

    results = collection.query(
        VectorQuery("embedding", vector=query_vector),
        topk=3,
    )

    print("Top-3 results:")
    print("-" * 60)
    for rank, result in enumerate(results, start=1):
        # Retrieve stored fields if available
        doc_id = result.id
        score = result.score
        # Find original text for display
        original = next((d for d in SAMPLE_DOCUMENTS if d["id"] == doc_id), {})
        print(f"  #{rank}  id={doc_id}  score={score:.4f}")
        print(f"       {original.get('text', '(text not stored)')}")
    print("-" * 60)

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #
    collection.destroy()
    print("\nCollection destroyed.  Done!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="zvec custom HTTP embedding demo (LM Studio / Ollama)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:1234",
        help="Base URL of the OpenAI-compatible embedding server.",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-nomic-embed-text-v1.5@f16",
        help="Embedding model name as expected by the server.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key (leave blank for local servers).",
    )
    parser.add_argument(
        "--collection-path",
        default="/tmp/zvec_http_embedding_example",
        help="Filesystem path for the zvec collection.",
    )
    args = parser.parse_args()

    run_demo(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        collection_path=args.collection_path,
    )


if __name__ == "__main__":
    main()
