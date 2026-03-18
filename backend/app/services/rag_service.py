"""
RAG (Retrieval Augmented Generation) service for document-based Q&A.
Backed by Qdrant Cloud vector store.
"""

from typing import List, Dict, Any, Optional
import logging
import asyncio
from functools import partial

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
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
            logger.info(f"Connecting to Qdrant Cloud: {settings.qdrant_url}")
            self.qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
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
        # LangChain QdrantVectorStore stores metadata at the TOP LEVEL of the
        # Qdrant payload (not nested under a 'metadata' key).
        # create_payload_index is idempotent — safe to call every startup.
        _filter_fields = [
            "device_type",
            "brand",
            "model",
            "document_id",
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
            chunks = []
            for doc, score in results:
                # Qdrant cosine similarity score is already in [0,1] (higher = more similar)
                relevance_score = float(score)

                if relevance_score >= settings.relevance_threshold:
                    chunks.append(
                        {
                            "content": doc.page_content,
                            "source_file": doc.metadata.get("source_file", "Unknown"),
                            "page_number": doc.metadata.get("page_number"),
                            "section_name": doc.metadata.get("section_name"),
                            "relevance_score": round(relevance_score, 3),
                            "device_type": doc.metadata.get("device_type"),
                            "brand": doc.metadata.get("brand"),
                            "model": doc.metadata.get("model"),
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
    ) -> int:
        """Add documents to the Qdrant vector store."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                partial(
                    self.vector_store.add_texts,
                    texts=texts,
                    metadatas=metadatas,
                ),
            )
            logger.info(f"Added {len(texts)} chunks to Qdrant")
            return len(texts)

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_document(self, document_id: str) -> bool:
        """Delete all vectors belonging to a document from Qdrant."""
        try:
            self.qdrant_client.delete(
                collection_name=settings.qdrant_collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
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

    def _build_filter(
        self,
        device_type: Optional[str],
        brand: Optional[str],
        model: Optional[str],
    ) -> Optional[Filter]:
        """Build a Qdrant Filter from optional metadata fields.

        LangChain's QdrantVectorStore stores metadata at the TOP LEVEL of the
        Qdrant payload, so the field keys are plain names (e.g. 'device_type'),
        NOT nested paths like 'metadata.device_type'.
        """
        conditions = []

        if device_type:
            conditions.append(
                FieldCondition(key="device_type", match=MatchValue(value=device_type))
            )
        if brand:
            conditions.append(
                FieldCondition(key="brand", match=MatchValue(value=brand))
            )
        if model:
            conditions.append(
                FieldCondition(key="model", match=MatchValue(value=model))
            )

        return Filter(must=conditions) if conditions else None


# Global RAG service instance
rag_service = RAGService()
