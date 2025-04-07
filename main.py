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
    affiliations: Optional[str] = None
    details: Optional[str] = None
    date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    location: Optional[str] = None
    newsType: Optional[str] = None
    session_title: Optional[str] = None
    session_text: Optional[str] = None
    category: Optional[str] = None
    subCategory: Optional[str] = None
    disclosure: Optional[str] = None
    disease: Optional[List[Dict[str, str]]] = None
    sponsor: Optional[str] = None

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
import requests
import json
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

app = FastAPI()

# Config model
class Config:
    def __init__(self):
        self.openai_api_key = "your-openai-api-key"
        self.pinecone_api_key = "your-pinecone-api-key"

def get_config():
    return Config()

def get_pinecone_client(config: Config = Depends(get_config)):
    return Pinecone(api_key=config.pinecone_api_key)

# Request and response models
class CreateEmbeddingsRequest(BaseModel):
    index_name: str
    dimension: int = 1536
    namespace: str = "default"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_type: str = "openai"
    documents: List[Dict[str, Any]]

class EmbeddingResponse(BaseModel):
    success: bool
    message: str
    document_ids: List[str]
    count: int

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
        all_chunks = []
        all_metadatas = []
        
        for doc in request.documents:
            # Format disease information
            disease_info = "N/A"
            if "disease" in doc and doc["disease"]:
                disease_names = [d.get("name", "") for d in doc["disease"]]
                disease_info = ', '.join(disease_names)
            
            # Create content to be embedded
            content = f"""
                Title: {doc.get('session_title', '') or doc.get('brief_title', '') or 'N/A'}
                Date: {doc.get('date', 'N/A')}
                Time: {doc.get('start_time', 'N/A')} - {doc.get('end_time', 'N/A')}
                Location: {doc.get('location', 'N/A')}
                Category: {doc.get('category', 'N/A')}
                SubCategory: {doc.get('sub_category', 'N/A')}
                Diseases: {disease_info}
                Sponsor: {doc.get('sponsor', 'N/A')}
                Session Text: {doc.get('session_text', 'N/A')}
                Disclosure: {doc.get('disclosures', 'N/A')}
                News Type: {doc.get('news_type', 'N/A')}
                Affiliation: {doc.get('affiliations', 'N/A')}
                Details: {doc.get('details', 'N/A')}
                Summary: {doc.get('summary', 'N/A')}
                Authors: {doc.get('authors', 'N/A')}
            """
            
            # Split into chunks
            chunks = text_splitter.split_text(content)
            
            # Create metadata for each chunk
            for chunk in chunks:
                doc_id = str(uuid.uuid4())
                document_ids.append(doc_id)
                all_chunks.append(chunk)
                
                # Create metadata with only the specified fields
                metadata = {
                    "source_id": doc.get("source_id", ""),
                    "session_id": doc.get("session_id", ""),
                    "trial_ids": doc.get("trial_ids", ""),
                    "abstract_number": doc.get("abstract_number", ""),
                    "Conf_Upload_Version": doc.get("Conf_Upload_Version", ""),
                    "date": doc.get("date", ""),
                    "start_time": doc.get("start_time", ""),
                    "end_time": doc.get("end_time", ""),
                    "location": doc.get("location", ""),
                    "news_type": doc.get("news_type", ""),
                    "category": doc.get("category", ""),
                    "sub_category": doc.get("sub_category", ""),
                    "document_id": doc_id
                }
                
                # Add disease information if available
                if "disease" in doc and doc["disease"]:
                    metadata["disease"] = doc["disease"]
                
                all_metadatas.append(metadata)
        
        # Initialize embedding model (only once)
        if request.embedding_type.lower() == "openai":
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-large", 
                openai_api_key=config.openai_api_key
            )
            
            # Process in batches
            batch_size = 100  # Adjust as needed
            total_chunks = len(all_chunks)
            
            for i in range(0, total_chunks, batch_size):
                batch_end = min(i + batch_size, total_chunks)
                print(f"Processing chunks {i} to {batch_end}")
                
                # Get batch of chunks and their IDs
                batch_chunks = all_chunks[i:batch_end]
                batch_ids = document_ids[i:batch_end]
                batch_metadatas = all_metadatas[i:batch_end]
                
                # Generate embeddings for the batch
                batch_embeddings = embedding_model.embed_documents(batch_chunks)
                
                # Prepare vectors for upsert
                vectors = []
                for j, embedding in enumerate(batch_embeddings):
                    vectors.append({
                        "id": batch_ids[j],
                        "values": embedding,
                        "metadata": batch_metadatas[j]
                    })
                
                # Upsert batch of vectors
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
class QueryRequest(BaseModel):
    index_name: str
    query_text: str
    top_k: int = 5
    embedding_type: str = "openai"
    namespace: Optional[str] = "default"
    include_metadata: bool = True
    filter: Optional[Dict[str, Any]] = None
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