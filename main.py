from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import json
import os
import uuid
import uvicorn
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

app = FastAPI(title="Vector Database API", 
              description="API for managing document embeddings in Pinecone")

# Environment configuration
class Config:
    def __init__(self):
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.pinecone_api_key or not self.openai_api_key:
            raise ValueError("Missing required API keys in environment variables")

# Dependency to get config
def get_config():
    return Config()

# Initialize Pinecone client
def get_pinecone_client(config: Config = Depends(get_config)):
    return Pinecone(api_key=config.pinecone_api_key)

# Models
class DocumentMetadata(BaseModel):
    manual_id: Optional[str] = ""
    source_id: Optional[str] = ""
    url: Optional[str] = ""
    date: Optional[str] = ""
    location: Optional[str] = ""
    category: Optional[str] = ""
    article_title: Optional[str] = ""
    session_title: Optional[str] = ""
    start_time: Optional[str] = ""
    end_time: Optional[str] = ""
    abstract_text: Optional[str] = ""

class Document(BaseModel):
    content: str
    metadata: DocumentMetadata

class CreateEmbeddingsRequest(BaseModel):
    index_name: str
    documents: List[Document]
    embedding_type: str = "openai"  # "openai" or "llama"
    dimension: int = 3072
    namespace: Optional[str] = "default"
    chunk_size: int = 2000
    chunk_overlap: int = 300

class UpsertEmbeddingsRequest(BaseModel):
    index_name: str
    documents: List[Document]
    embedding_type: str = "openai"
    namespace: Optional[str] = "default"

class DeleteEmbeddingsRequest(BaseModel):
    index_name: str
    ids: List[str]
    namespace: Optional[str] = "default"

class QueryRequest(BaseModel):
    index_name: str
    query_text: str
    top_k: int = 5
    embedding_type: str = "openai"
    namespace: Optional[str] = "default"
    include_metadata: bool = True
    filter: Optional[Dict[str, Any]] = None

class EmbeddingResponse(BaseModel):
    success: bool
    message: str
    document_ids: Optional[List[str]] = None
    count: Optional[int] = None

class QueryResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    results: List[QueryResult]
    count: int
    query_time_ms: Optional[float] = None

# 1. Create Embeddings API
@app.post("/api/create_embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: CreateEmbeddingsRequest,
    pc: Pinecone = Depends(get_pinecone_client),
    config: Config = Depends(get_config)
):
    """
    Create embeddings for documents and store them in Pinecone.
    This will create a new index if it doesn't exist.
    """
    try:
        # Create index if it doesn't exist
        if not pc.has_index(request.index_name):
            pc.create_index(
                name=request.index_name,
                dimension=request.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        # Get index
        index = pc.Index(request.index_name)
        
        # Process documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size, 
            chunk_overlap=request.chunk_overlap
        )
        
        document_ids = []
        vectors = []
        
        for doc in request.documents:
            # Create formatted content
            content = (
                f"Title: {doc.metadata.article_title}\n"
                f"Session: {doc.metadata.session_title}\n"
                f"Abstract: {doc.metadata.abstract_text}\n"
                f"Category: {doc.metadata.category}\n"
                f"Session Date: {doc.metadata.date}\n"
                f"Location: {doc.metadata.location}\n"
                f"Start Time: {doc.metadata.start_time}\n"
                f"End Time: {doc.metadata.end_time}\n"
                f"{doc.content}"
            )
            
            # Split into chunks
            chunks = text_splitter.split_text(content)
            
            # Process each chunk
            for chunk in chunks:
                doc_id = str(uuid.uuid4())
                document_ids.append(doc_id)
                
                # Generate embeddings based on selected method
                if request.embedding_type.lower() == "openai":
                    embedding_model = OpenAIEmbeddings(
                        model="text-embedding-3-large", 
                        openai_api_key=config.openai_api_key
                    )
                    embedding = embedding_model.embed_query(chunk)
                    
                    vectors.append({
                        "id": doc_id,
                        "values": embedding,
                        "metadata": {
                            **doc.metadata.dict(),
                            "content": chunk
                        }
                    })
                    
                elif request.embedding_type.lower() == "llama":
                    embedding_response = pc.inference.embed(
                        model="llama-text-embed-v2",
                        inputs=[chunk],
                        parameters={"input_type": "passage", "truncate": "END", "dimension": 2048},
                    )
                    
                    vectors.append({
                        "id": doc_id,
                        "values": embedding_response[0]['values'],
                        "metadata": {
                            **doc.metadata.dict(),
                            "content": chunk
                        }
                    })
        
        # Upsert vectors
        index.upsert(
            vectors=vectors,
            namespace=request.namespace
        )
        
        return EmbeddingResponse(
            success=True,
            message=f"Successfully created embeddings for {len(document_ids)} chunks from {len(request.documents)} documents",
            document_ids=document_ids,
            count=len(document_ids)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Upsert Embeddings API
@app.post("/api/upsert_embeddings", response_model=EmbeddingResponse)
async def upsert_embeddings(
    request: UpsertEmbeddingsRequest,
    pc: Pinecone = Depends(get_pinecone_client),
    config: Config = Depends(get_config)
):
    """
    Generate embeddings for documents and upsert them to an existing Pinecone index.
    """
    try:
        # Check if index exists
        if not pc.has_index(request.index_name):
            raise HTTPException(status_code=404, detail=f"Index {request.index_name} does not exist")
        
        # Get index
        index = pc.Index(request.index_name)
        
        # Process documents
        document_ids = []
        vectors = []
        
        for doc in request.documents:
            doc_id = str(uuid.uuid4())
            document_ids.append(doc_id)
            
            # Generate embeddings based on selected method
            if request.embedding_type.lower() == "openai":
                embedding_model = OpenAIEmbeddings(
                    model="text-embedding-3-large", 
                    openai_api_key=config.openai_api_key
                )
                embedding = embedding_model.embed_query(doc.content)
                
                vectors.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": {
                        **doc.metadata.dict(),
                        "content": doc.content
                    }
                })
                
            elif request.embedding_type.lower() == "llama":
                embedding_response = pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=[doc.content],
                    parameters={"input_type": "passage", "truncate": "END", "dimension": 2048},
                )
                
                vectors.append({
                    "id": doc_id,
                    "values": embedding_response[0]['values'],
                    "metadata": {
                        **doc.metadata.dict(),
                        "content": doc.content
                    }
                })
        
        # Upsert vectors
        index.upsert(
            vectors=vectors,
            namespace=request.namespace
        )
        
        return EmbeddingResponse(
            success=True,
            message=f"Successfully upserted {len(document_ids)} documents",
            document_ids=document_ids,
            count=len(document_ids)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Delete Embeddings API
@app.post("/api/delete_embeddings", response_model=EmbeddingResponse)
async def delete_embeddings(
    request: DeleteEmbeddingsRequest,
    pc: Pinecone = Depends(get_pinecone_client)
):
    """
    Delete embeddings from Pinecone by ID.
    """
    try:
        # Check if index exists
        if not pc.has_index(request.index_name):
            raise HTTPException(status_code=404, detail=f"Index {request.index_name} does not exist")
        
        # Get index
        index = pc.Index(request.index_name)
        
        # Delete vectors
        index.delete(
            ids=request.ids,
            namespace=request.namespace
        )
        
        return EmbeddingResponse(
            success=True,
            message=f"Successfully deleted {len(request.ids)} embeddings",
            count=len(request.ids)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. Query API
@app.post("/api/query", response_model=QueryResponse)
async def query_embeddings(
    request: QueryRequest,
    pc: Pinecone = Depends(get_pinecone_client),
    config: Config = Depends(get_config)
):
    """
    Query Pinecone for similar documents based on a text query.
    """
    try:
        # Check if index exists
        if not pc.has_index(request.index_name):
            raise HTTPException(status_code=404, detail=f"Index {request.index_name} does not exist")
        
        # Get index
        index = pc.Index(request.index_name)
        
        # Generate query embedding
        if request.embedding_type.lower() == "openai":
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-large", 
                openai_api_key=config.openai_api_key
            )
            query_embedding = embedding_model.embed_query(request.query_text)
            
        elif request.embedding_type.lower() == "llama":
            embedding_response = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[request.query_text],
                parameters={"input_type": "query", "truncate": "END", "dimension": 2048},
            )
            query_embedding = embedding_response[0]['values']
        
        # Query the index
        query_response = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=request.include_metadata,
            namespace=request.namespace,
            filter=request.filter
        )
        
        # Format results
        results = []
        for match in query_response.matches:
            results.append(QueryResult(
                id=match.id,
                score=match.score,
                metadata=match.metadata
            ))
        
        return QueryResponse(
            results=results,
            count=len(results),
            query_time_ms=query_response.usage.total_ms if hasattr(query_response, 'usage') else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper endpoint to check existing indexes
@app.get("/api/indexes", response_model=List[str])
async def list_indexes(pc: Pinecone = Depends(get_pinecone_client)):
    """List all available indexes in Pinecone."""
    return pc.list_indexes()

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)