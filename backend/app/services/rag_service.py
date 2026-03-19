"""
RAG (Retrieval Augmented Generation) service for document-based Q&A.
Backed by Qdrant Cloud vector store.
"""

from typing import List, Dict, Any, Optional
import logging
import asyncio
import time
from functools import partial

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG-based document retrieval and question answering."""

    def __init__(self):
        """Initialize RAG service with embeddings and Qdrant vector store."""
        self.embeddings = None
        self.vector_store = None
        self.qdrant_client = None
        self.llm = None
        self.text_splitter = None
        self.initialized = False
        try:
            self._initialize()
            self.initialized = True
        except Exception as e:
            logger.error(
                f"RAG service failed to initialise (auth/chat may still work): {e}"
            )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self):
        """Initialize all components."""
        try:
            # Embedding model
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # Text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )

            # Qdrant Cloud client
            # timeout=120: the default (~5s) was too short — the upsert HTTP
            # request was being killed before Qdrant could respond.
            logger.info(f"Connecting to Qdrant Cloud: {settings.qdrant_url}")
            self.qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=120,
            )

            # Ensure collection exists
            self._ensure_collection()

            # LangChain Qdrant vector store
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=settings.qdrant_collection_name,
                embedding=self.embeddings,
            )

            # LLM
            if settings.llm_provider == "openai":
                self.llm = ChatOpenAI(
                    model_name=settings.llm_model,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    openai_api_key=settings.openai_api_key,
                )
            elif settings.llm_provider == "google":
                self.llm = ChatGoogleGenerativeAI(
                    model=settings.llm_model,
                    temperature=settings.llm_temperature,
                    google_api_key=settings.google_api_key,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

            logger.info("RAG service initialised successfully with Qdrant Cloud")

        except Exception as e:
            logger.error(f"Failed to initialise RAG service: {e}")
            raise  # Re-raised so __init__ can catch and mark initialized=False

    def _ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist yet, and ensure
        keyword payload indexes exist on all filterable metadata fields."""
        collections = [c.name for c in self.qdrant_client.get_collections().collections]
        if settings.qdrant_collection_name not in collections:
            logger.info(
                f"Creating Qdrant collection: {settings.qdrant_collection_name}"
            )
            self.qdrant_client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
        else:
            logger.info(
                f"Qdrant collection '{settings.qdrant_collection_name}' already exists"
            )

        # Ensure keyword indexes exist for all filterable metadata fields.
        # Two schemas coexist in the collection:
        #   Schema A (PDF uploads)   : flat top-level keys  e.g.  device_type
        #   Schema B (ChromaDB mig.) : nested under metadata e.g.  metadata.device_type
        # We index BOTH paths so Qdrant can filter efficiently on either.
        # create_payload_index is idempotent — safe to call every startup.
        _filter_fields = [
            "device_type",
            "brand",
            "model",
            "document_id",
            # nested paths for ChromaDB-migrated points
            "metadata.device_type",
            "metadata.brand",
            "metadata.model",
            "metadata.document_id",
        ]
        for field in _filter_fields:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=settings.qdrant_collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info(f"Payload index ensured for field: {field}")
            except Exception as idx_err:
                # Index may already exist with different schema — log and continue.
                logger.warning(f"Could not create payload index for '{field}': {idx_err}")

    # ------------------------------------------------------------------
    # Prompt template
    # ------------------------------------------------------------------

    def create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for troubleshooting."""
        template = """You are an expert device repair technician with years of experience troubleshooting various household appliances and electronics.

Using the manual excerpts provided below, provide clear, step-by-step troubleshooting instructions for the user's problem.

Context from device manuals:
{context}

User Question: {question}

Instructions for your response:
1. Start by diagnosing the most likely cause of the problem
2. Provide clear, numbered step-by-step troubleshooting instructions
3. Include any relevant safety warnings (electrical hazards, water damage risks, etc.)
4. Cite the specific manual section you're referencing
5. If the problem requires professional repair, clearly state this
6. If the provided context doesn't contain relevant information, honestly say "I don't have specific information about this in the available manuals" and provide general guidance if appropriate

Important:
- Be conversational and friendly, but professional
- Use simple language, avoiding technical jargon when possible
- If you use technical terms, briefly explain them
- Prioritize user safety above all else

Your response:"""

        return PromptTemplate(template=template, input_variables=["context", "question"])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve_relevant_chunks(
        self,
        query: str,
        device_type: Optional[str] = None,
        brand: Optional[str] = None,
        model: Optional[str] = None,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query using Qdrant filters."""
        if top_k is None:
            top_k = settings.retrieval_top_k

        try:
            # Build Qdrant filter from optional metadata fields
            qdrant_filter = self._build_filter(device_type=device_type, brand=brand, model=model)

            # Similarity search (run blocking call in executor)
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None,
                partial(
                    self.vector_store.similarity_search_with_score,
                    query,
                    k=top_k,
                    filter=qdrant_filter,
                ),
            )

            # Format results
            # NOTE: Two payload schemas coexist:
            #   Schema A (PDF uploads)   – flat top-level keys
            #   Schema B (ChromaDB mig.) – fields nested under a 'metadata' dict
            # _get_meta() resolves a field from either location transparently.
            chunks = []
            for doc, score in results:
                # Qdrant cosine similarity score is already in [0,1] (higher = more similar)
                relevance_score = float(score)

                if relevance_score >= settings.relevance_threshold:
                    chunks.append(
                        {
                            "content": doc.page_content,
                            "source_file": self._get_meta(doc.metadata, "source_file", "Unknown"),
                            "page_number": self._get_meta(doc.metadata, "page_number"),
                            "section_name": self._get_meta(doc.metadata, "section_name"),
                            "relevance_score": round(relevance_score, 3),
                            "device_type": self._get_meta(doc.metadata, "device_type"),
                            "brand": self._get_meta(doc.metadata, "brand"),
                            "model": self._get_meta(doc.metadata, "model"),
                        }
                    )

            logger.info(f"Retrieved {len(chunks)} relevant chunks for query: {query[:50]}...")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    async def generate_answer(
        self,
        query: str,
        device_type: Optional[str] = None,
        brand: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an answer using RAG."""
        try:
            chunks = await self.retrieve_relevant_chunks(
                query=query,
                device_type=device_type,
                brand=brand,
                model=model,
            )

            if not chunks:
                return {
                    "answer": (
                        "I don't have specific information about this issue in the available manuals. "
                        "I recommend checking the device's official manual or contacting customer support."
                    ),
                    "sources": [],
                }

            context = "\n\n".join(
                [
                    f"[Source: {chunk['source_file']}, Page: {chunk.get('page_number', 'N/A')}]\n{chunk['content']}"
                    for chunk in chunks
                ]
            )

            prompt_template = self.create_prompt_template()
            prompt = prompt_template.format(context=context, question=query)

            response = await self.llm.ainvoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            sources = [
                {
                    "content": chunk["content"][:200] + "..."
                    if len(chunk["content"]) > 200
                    else chunk["content"],
                    "source_file": chunk["source_file"],
                    "page_number": chunk.get("page_number"),
                    "section_name": chunk.get("section_name"),
                    "relevance_score": chunk["relevance_score"],
                }
                for chunk in chunks[:5]
            ]

            return {"answer": answer, "sources": sources}

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 25,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> int:
        """Add documents to the Qdrant vector store in small batches with retries.

        Two-phase upload per batch:
          1. Embed texts on CPU  (no Qdrant connection held open – no timeout risk)
          2. Upsert pre-computed vectors to Qdrant  (fast network call)

        If the upsert step times out, it is retried up to `max_retries` times
        with a `retry_delay`-second pause.  Embeddings are cached so we never
        recompute them on a retry.
        """
        total = len(texts)
        added = 0
        loop = asyncio.get_running_loop()

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]
            batch_metas = metadatas[start:end]
            batch_num = start // batch_size + 1

            # ── Step 1: Embed texts (CPU-bound; runs once per batch) ───────
            logger.info(f"Embedding batch {batch_num} ({start}-{end-1} of {total})…")
            embeddings_list: List[List[float]] = await loop.run_in_executor(
                None,
                partial(self.embeddings.embed_documents, batch_texts),
            )

            # ── Step 2: Build PointStructs ─────────────────────────────
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "page_content": text,
                        "metadata": meta,
                    },
                )
                for text, meta, embedding in zip(
                    batch_texts, batch_metas, embeddings_list
                )
            ]

            # ── Step 3: Upload to Qdrant (with retry on timeout) ─────────
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    await loop.run_in_executor(
                        None,
                        partial(
                            self.qdrant_client.upsert,
                            collection_name=settings.qdrant_collection_name,
                            points=points,
                        ),
                    )
                    added += len(batch_texts)
                    logger.info(
                        f"Batch {batch_num} uploaded ✓  "
                        f"({added}/{total} chunks done)"
                        + (f"  [attempt {attempt}]" if attempt > 1 else "")
                    )
                    last_error = None
                    break  # success — move to next batch
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Batch {batch_num} attempt {attempt} failed: {e}. "
                            f"Retrying in {retry_delay}s…"
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(
                            f"Batch {batch_num} failed after {max_retries} attempts: {e}"
                        )

            if last_error is not None:
                raise last_error  # propagate only after all retries exhausted

        logger.info(f"Successfully stored all {added} chunks in Qdrant")
        return added

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_document(self, document_id: str) -> bool:
        """Delete all vectors belonging to a document from Qdrant.

        Handles both payload schemas:
          Schema A (PDF uploads)   – document_id at top level
          Schema B (ChromaDB mig.) – document_id nested under metadata
        """
        try:
            self.qdrant_client.delete(
                collection_name=settings.qdrant_collection_name,
                points_selector=Filter(
                    should=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        ),
                        FieldCondition(
                            key="metadata.document_id",
                            match=MatchValue(value=document_id),
                        ),
                    ]
                ),
            )
            logger.info(f"Deleted document {document_id} from Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_meta(self, metadata: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Read a metadata field from either the flat schema (PDF uploads) or the
        nested schema (ChromaDB-migrated points where fields live under 'metadata').
        Flat schema takes priority if both are present.
        """
        # Flat schema (Schema A – PDF uploads)
        if key in metadata and metadata[key] is not None:
            return metadata[key]
        # Nested schema (Schema B – ChromaDB migration)
        nested = metadata.get("metadata") or {}
        if isinstance(nested, dict) and key in nested and nested[key] is not None:
            return nested[key]
        return default

    def _build_filter(
        self,
        device_type: Optional[str],
        brand: Optional[str],
        model: Optional[str],
    ) -> Optional[Filter]:
        """Build a Qdrant Filter from optional metadata fields.

        Two payload schemas coexist in the collection:
          Schema A (PDF uploads)   – flat keys:            device_type, brand, model
          Schema B (ChromaDB mig.) – nested under metadata: metadata.device_type, …

        For each provided filter value we create a 'should' (OR) sub-filter that
        matches either schema, then wrap all sub-filters in a 'must' (AND) block
        so that multiple filters are applied together.
        """
        must_conditions = []

        if device_type:
            must_conditions.append(
                Filter(should=[
                    FieldCondition(key="device_type",          match=MatchValue(value=device_type)),
                    FieldCondition(key="metadata.device_type", match=MatchValue(value=device_type)),
                ])
            )
        if brand:
            must_conditions.append(
                Filter(should=[
                    FieldCondition(key="brand",          match=MatchValue(value=brand)),
                    FieldCondition(key="metadata.brand", match=MatchValue(value=brand)),
                ])
            )
        if model:
            must_conditions.append(
                Filter(should=[
                    FieldCondition(key="model",          match=MatchValue(value=model)),
                    FieldCondition(key="metadata.model", match=MatchValue(value=model)),
                ])
            )

        return Filter(must=must_conditions) if must_conditions else None


# Global RAG service instance
rag_service = RAGService()
