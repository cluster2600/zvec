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
"""LangChain VectorStore integration for zvec.

This module provides :class:`ZvecVectorStore`, a LangChain-compatible
``VectorStore`` backed by the zvec embedded vector database.

Optional dependency
-------------------
This module requires ``langchain-core``::

    pip install langchain-core

zvec itself must also be installed::

    pip install zvec

Example usage
-------------
.. code-block:: python

    from zvec.langchain.vectorstore import ZvecVectorStore
    from langchain_core.embeddings import FakeEmbeddings

    embeddings = FakeEmbeddings(size=384)
    store = ZvecVectorStore.from_texts(
        texts=["Hello world", "Vector search with zvec"],
        embedding=embeddings,
        path="./my_zvec_store",
    )
    docs = store.similarity_search("hello", k=2)
    for doc in docs:
        print(doc.page_content)
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Iterable, List, Optional, Tuple

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGCHAIN_AVAILABLE = False
    # Provide stub base class so the module can still be imported without
    # langchain_core installed; a RuntimeError is raised at instantiation.
    class VectorStore:  # type: ignore[no-redef]
        pass

    class Embeddings:  # type: ignore[no-redef]
        pass

    class Document:  # type: ignore[no-redef]
        pass


import zvec
from zvec.typing import DataType

__all__ = ["ZvecVectorStore"]

# Fixed field/vector names used in the underlying zvec collection.
_FIELD_TEXT = "text"
_FIELD_METADATA = "metadata"
_VECTOR_FIELD = "embedding"


def _check_langchain() -> None:
    """Raise a helpful error when langchain_core is not installed."""
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain-core is required to use ZvecVectorStore. "
            "Install it with: pip install langchain-core"
        )


class ZvecVectorStore(VectorStore):
    """A LangChain ``VectorStore`` backed by zvec.

    ``ZvecVectorStore`` persists documents and their embeddings in a local
    zvec collection. It is compatible with the full LangChain retrieval
    ecosystem, including chains, agents, and retrievers.

    The collection schema is created automatically on first use and contains:

    * ``text`` — the raw document content (``STRING``)
    * ``metadata`` — JSON-serialised document metadata (``STRING``)
    * ``embedding`` — the dense vector embedding (``VECTOR_FP32``)

    Args:
        embedding (Embeddings): A LangChain ``Embeddings`` object used to
            convert texts to vectors.
        path (str): Filesystem path where the zvec collection is stored.
        dimension (int): Dimensionality of the embedding vectors.  Must match
            the output size of *embedding*. Defaults to ``1536``.
        collection (zvec.Collection, optional): An already-opened zvec
            ``Collection``.  If ``None`` (default), a new collection is
            created at *path*.

    Note:
        ``langchain-core`` is an optional dependency.  Install it with::

            pip install langchain-core

    Raises:
        ImportError: If ``langchain-core`` is not installed.

    Examples:
        >>> from zvec.langchain.vectorstore import ZvecVectorStore
        >>> store = ZvecVectorStore.from_texts(
        ...     texts=["Hello", "World"],
        ...     embedding=my_embeddings,
        ...     path="./store",
        ... )
        >>> results = store.similarity_search("hello", k=1)
    """

    def __init__(
        self,
        embedding: Embeddings,
        path: str,
        dimension: int = 1536,
        collection: Optional[zvec.Collection] = None,
    ) -> None:
        _check_langchain()

        self._embedding = embedding
        self._path = path
        self._dimension = dimension

        if collection is not None:
            self._collection = collection
        else:
            self._collection = self._create_collection(path, dimension)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_collection(path: str, dimension: int) -> zvec.Collection:
        """Create (or open) a zvec collection with the standard LangChain schema."""
        schema = zvec.CollectionSchema(
            name="langchain_store",
            fields=[
                zvec.FieldSchema(_FIELD_TEXT, DataType.STRING, nullable=False),
                zvec.FieldSchema(_FIELD_METADATA, DataType.STRING, nullable=True),
            ],
            vectors=[
                zvec.VectorSchema(
                    _VECTOR_FIELD,
                    DataType.VECTOR_FP32,
                    dimension=dimension,
                    index_param=zvec.HnswIndexParam(metric=zvec.MetricType.COSINE),
                ),
            ],
        )
        try:
            return zvec.create_and_open(path=path, schema=schema)
        except Exception:
            # Collection already exists — open it instead.
            return zvec.open(path=path)

    def _texts_to_docs(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[zvec.Doc]:
        """Convert texts + embeddings to a list of :class:`zvec.Doc` objects."""
        text_list = list(texts)
        if metadatas is None:
            metadatas = [{} for _ in text_list]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in text_list]

        docs = []
        for doc_id, text, meta, vector in zip(ids, text_list, metadatas, embeddings):
            docs.append(
                zvec.Doc(
                    id=doc_id,
                    fields={
                        _FIELD_TEXT: text,
                        _FIELD_METADATA: json.dumps(meta, ensure_ascii=False),
                    },
                    vectors={_VECTOR_FIELD: vector},
                )
            )
        return docs

    @staticmethod
    def _doc_to_langchain(doc: zvec.Doc) -> Document:
        """Convert a zvec :class:`Doc` to a LangChain :class:`Document`."""
        text = doc.field(_FIELD_TEXT) or ""
        raw_meta = doc.field(_FIELD_METADATA) or "{}"
        try:
            metadata = json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return Document(page_content=text, metadata=metadata)

    # ------------------------------------------------------------------
    # LangChain VectorStore interface
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """The underlying :class:`langchain_core.embeddings.Embeddings` object."""
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed and insert a batch of texts into the collection.

        Args:
            texts (Iterable[str]): Texts to add.
            metadatas (Optional[List[dict]]): Optional metadata dicts, one per
                text.  Missing entries are replaced with ``{}``.
            ids (Optional[List[str]]): Optional document IDs.  UUIDs are
                generated when omitted.
            **kwargs: Additional keyword arguments (reserved for future use).

        Returns:
            List[str]: The document IDs assigned to the inserted texts.
        """
        text_list = list(texts)
        if not text_list:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in text_list]

        embeddings = self._embedding.embed_documents(text_list)
        docs = self._texts_to_docs(text_list, embeddings, metadatas, ids)
        self._collection.upsert(docs)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return the *k* most similar documents to *query*.

        Args:
            query (str): The query text.
            k (int): Number of results to return. Defaults to ``4``.
            filter (Optional[str]): Optional zvec filter expression
                (e.g. ``"category == 'science'"``).
            **kwargs: Additional keyword arguments forwarded to
                :meth:`zvec.Collection.query`.

        Returns:
            List[Document]: Matching LangChain documents ordered by similarity.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _score in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return the *k* most similar documents together with their scores.

        A higher score indicates greater similarity (cosine distance is
        converted to cosine similarity internally by zvec).

        Args:
            query (str): The query text.
            k (int): Number of results to return. Defaults to ``4``.
            filter (Optional[str]): Optional zvec filter expression.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`zvec.Collection.query`.

        Returns:
            List[Tuple[Document, float]]: List of ``(document, score)`` pairs
            ordered from most to least similar.
        """
        query_vector = self._embedding.embed_query(query)
        vector_query = zvec.VectorQuery(_VECTOR_FIELD, vector=query_vector)

        results: List[zvec.Doc] = self._collection.query(
            vectors=vector_query,
            topk=k,
            filter=filter,
            **kwargs,
        )

        return [
            (self._doc_to_langchain(doc), doc.score if doc.score is not None else 0.0)
            for doc in results
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        path: str = "./zvec_store",
        dimension: Optional[int] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "ZvecVectorStore":
        """Create a :class:`ZvecVectorStore` from a list of texts.

        This is the primary factory method for building a store from an
        existing corpus of documents.  It embeds *texts*, creates the zvec
        collection at *path*, inserts all documents, and returns a ready-to-use
        store instance.

        Args:
            texts (List[str]): Texts to embed and insert.
            embedding (Embeddings): LangChain ``Embeddings`` implementation.
            metadatas (Optional[List[dict]]): Optional metadata, one dict per
                text.
            path (str): Filesystem path for the zvec collection.
                Defaults to ``"./zvec_store"``.
            dimension (Optional[int]): Embedding dimensionality.  When
                ``None`` (default), the dimension is inferred by embedding the
                first text (or falls back to ``1536``).
            ids (Optional[List[str]]): Optional document IDs.
            **kwargs: Extra keyword arguments forwarded to the constructor.

        Returns:
            ZvecVectorStore: An initialised and populated store.

        Examples:
            >>> store = ZvecVectorStore.from_texts(
            ...     texts=["Hello world", "Python rocks"],
            ...     embedding=my_embeddings,
            ...     path="./my_store",
            ... )
        """
        _check_langchain()

        # Infer dimension from first embedding when not provided.
        if dimension is None:
            if texts:
                sample = embedding.embed_query(texts[0])
                dimension = len(sample)
            else:
                dimension = 1536  # safe default

        store = cls(embedding=embedding, path=path, dimension=dimension, **kwargs)
        if texts:
            store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
