import os
import json
import uuid
import requests
import time
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("vector_db_api.log")  # Log to file
    ]
)
logger = logging.getLogger("vector_db_api")

app = FastAPI(title="Vector Database API", 
              description="API for managing document embeddings in Pinecone")

# Environment configuration
class Config:
    def __init__(self):
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.larvol_api_key = os.environ.get("LARVOL_API_KEY")  # Add this if needed
        
        # Log configuration status (without exposing keys)
        if not self.pinecone_api_key:
            logger.error("PINECONE_API_KEY environment variable not set")
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            
        if not self.pinecone_api_key or not self.openai_api_key:
            raise ValueError("Missing required API keys in environment variables")
        
        logger.info("Configuration initialized successfully")

# Dependency to get config
def get_config():
    return Config()

# Initialize Pinecone client
def get_pinecone_client(config: Config = Depends(get_config)):
    logger.info("Initializing Pinecone client")
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
    document_ids: List[str] = []
    count: int = 0
    job_id: Optional[str] = None

class ConferenceDataRequest(BaseModel):
    conferenceName: str
    conferenceIteration: str
    index_name: str = "conference-data"
    namespace: str = "default"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_pages: int = 80  # Safety limit for page fetching

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    processed_items: int = 0
    total_items: int = 0
    current_page: int = 0
    total_pages: int = 0
    message: Optional[str] = None

# Dictionary to track background jobs
background_jobs = {}

def fetch_conference_data(conference_name: str, page: int = 1):
    """Fetch conference data from the API for a specific page"""
    api_url = f"https://lt.larvol.com/api/news.php?paged_mode=yes&page={page}&conf_name={conference_name}"
    
    logger.info(f"Fetching data from {api_url}")
    
    # Get API key from environment
    api_key = "596c36c6-0ecd-11f0-8583-0221c24e933b"
    
    if not api_key:
        logger.error("LARVOL_API_KEY environment variable is not set")
        raise ValueError("LARVOL_API_KEY environment variable is not set")
    
    logger.info(f"Using API key: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}")
    
    # Try different authentication methods
    try:
        # Method 1: Bearer token in Authorization header
        headers1 = {
            'Authorization': '596c36c6-0ecd-11f0-8583-0221c24e933b'
            }
        logger.info("Trying authentication method 1: Bearer token in Authorization header")
        
        response = requests.request("GET", api_url, headers=headers1, data={})
        logger.info(f"Response from thje API: {response}")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Method 1 succeeded. Response data length: {len(data)}")
            
            # Validate response format
            if isinstance(data, list) and len(data) >= 2:
                # Get total pages from the first item (if available)
                total_pages = 1
                if "total_pages" in data[0]:
                    try:
                        total_pages = int(data[0]["total_pages"])
                        logger.info(f"Total pages detected from API: {total_pages}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error parsing total_pages: {e}")
                
                # Remove the metadata element (first item) and return the actual data
                return data[1:], total_pages
            else:
                # If response is valid JSON but not in the expected format, log it
                logger.error(f"Response not in expected format: {data}")
                
                # The API might be returning an error message with different structure
                if isinstance(data, dict) and "result" in data and data["result"] == "error":
                    error_msg = data.get("message", "Unknown API error")
                    logger.error(f"API returned error: {error_msg}")
                    raise ValueError(f"API error: {error_msg}")
        else:
            logger.warning(f"Method 1 failed with status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Method 1 failed: {e}")
        
def prepare_safe_metadata(doc, doc_id, conference_name=None):
    """
    Prepare metadata that is safe for Pinecone.
    Converts None values to empty strings and handles other invalid types.
    """
    # Format disease information
    disease_info = ""
    if "disease" in doc and doc["disease"] and isinstance(doc["disease"], list):
        disease_names = [d.get("name", "") for d in doc["disease"] if d.get("name")]
        disease_info = ', '.join(disease_names)
    
    title = doc.get('session_title', '') or doc.get('brief_title', '') or ''
    
    # Create base metadata with default empty strings for None values
    metadata = {
        "title": title or "",
        "source_id": str(doc.get("source_id", "")) if doc.get("source_id") is not None else "",
        "session_id": str(doc.get("session_id", "")) if doc.get("session_id") is not None else "",
        "abstract_number": str(doc.get("abstract_number", "")) if doc.get("abstract_number") is not None else "",
        "date": str(doc.get("date", "")) if doc.get("date") is not None else "",
        "start_time": str(doc.get("start_time", "")) if doc.get("start_time") is not None else "",
        "end_time": str(doc.get("end_time", "")) if doc.get("end_time") is not None else "",
        "location": str(doc.get("location", "")) if doc.get("location") is not None else "",
        "news_type": str(doc.get("news_type", "")) if doc.get("news_type") is not None else "",
        "category": str(doc.get("category", "")) if doc.get("category") is not None else "",
        "sub_category": str(doc.get("sub_category", "")) if doc.get("sub_category") is not None else "",
        "document_id": doc_id,
    }
    
    # Add conference name if provided
    if conference_name:
        metadata["conference_name"] = str(conference_name)
    
    # Add disease information if available
    if disease_info:
        metadata["disease_names"] = disease_info
    
    # Handle trial_ids specifically, as mentioned in the error
    if doc.get("trial_ids") is not None and doc.get("trial_ids") != "null":
        metadata["trial_ids"] = str(doc.get("trial_ids"))
    
    # Add other fields if they exist and aren't None
    optional_fields = [
        "authors", "details", "summary", "sponsor", "affiliations", 
        "Conf_Upload_Version", "session_text"
    ]
    
    for field in optional_fields:
        if field in doc and doc[field] is not None:
            metadata[field] = str(doc[field])
    
    return metadata

async def process_conference_data(
    conference_name: str,
    conference_iteration: str,
    index_name: str,
    namespace: str,
    chunk_size: int,
    chunk_overlap: int,
    max_pages: int,
    job_id: str,
    pc: Pinecone,
    config: Config
):
    """Background task to fetch conference data and create embeddings"""
    try:
        logger.info(f"Starting job {job_id} for conference {conference_name} {conference_iteration}")
        
        background_jobs[job_id] = {
            "status": "running", 
            "processed_items": 0, 
            "total_items": 0,
            "current_page": 0,
            "total_pages": 0
        }
        # max_pages = 3
        # Format the conference name for the API
        formatted_conf_name = f"{conference_name} {conference_iteration}"
        logger.info(f"Formatted conference name: {formatted_conf_name}")
        
        # Fetch first page to get total pages
        logger.info(f"Fetching first page to determine total pages")
        documents, total_pages = fetch_conference_data(formatted_conf_name, 1)
        print(f"Total pages: {total_pages},Max pages: {max_pages}")
        # Limit total pages to max_pages
        total_pages = min(total_pages, max_pages)
        logger.info(f"Will process {total_pages} pages (limited by max_pages={max_pages})")
        
        background_jobs[job_id]["total_pages"] = total_pages
        background_jobs[job_id]["status"] = f"Fetching data from {total_pages} pages"
        
        # Collect all documents from all pages
        all_documents = documents.copy()
        logger.info(f"Collected {len(documents)} documents from page 1")
        
        for page in range(2, total_pages + 1):
            background_jobs[job_id]["current_page"] = page
            background_jobs[job_id]["status"] = f"Fetching page {page} of {total_pages}"
            logger.info(f"Fetching page {page} of {total_pages}")
            
            try:
                page_documents, _ = fetch_conference_data(formatted_conf_name, page)
                logger.info(f"Collected {len(page_documents)} documents from page {page}")
                all_documents.extend(page_documents)
                
                # Add a short delay to avoid overwhelming the API
                time.sleep(0.5)
            except Exception as e:
                error_msg = f"Error fetching page {page}: {str(e)}"
                logger.error(error_msg)
                background_jobs[job_id]["message"] = error_msg
                # Continue with the next page
        
        # Update job status
        total_document_count = len(all_documents)
        background_jobs[job_id]["total_items"] = total_document_count
        background_jobs[job_id]["status"] = f"Processing {total_document_count} documents"
        logger.info(f"Total documents collected: {total_document_count}")
        
        # Create index if it doesn't exist
        logger.info(f"Checking if index {index_name} exists")
        if not pc.has_index(index_name):
            logger.info(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=3072,  # Using dimension=1536 for text-embedding-3-large
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            logger.info(f"Index {index_name} already exists")
        
        # Get index
        index = pc.Index(index_name)
        
        # Process documents
        logger.info(f"Initializing text splitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Initialize embedding model
        logger.info("Initializing OpenAI embedding model (text-embedding-3-large)")
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large", 
            openai_api_key=config.openai_api_key
        )
        
        document_ids = []
        processed_count = 0
        
        # Process in batches
        batch_size = 20  # Adjust based on API rate limits and memory constraints
        logger.info(f"Processing documents in batches of {batch_size}")
        
        for i in range(0, len(all_documents), batch_size):
            batch_docs = all_documents[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(all_documents) + batch_size - 1)//batch_size} ({len(batch_docs)} documents)")
            
            all_chunks = []
            all_metadatas = []
            chunk_ids = []
            
            for doc_idx, doc in enumerate(batch_docs):
                # Format disease information
                disease_info = "N/A"
                if "disease" in doc and doc["disease"]:
                    disease_names = [d.get("name", "") for d in doc["disease"] if d.get("name")]
                    disease_info = ', '.join(disease_names)
                
                # Create content to be embedded - select the most relevant fields
                title = doc.get('session_title', '') or doc.get('brief_title', '') or 'N/A'
                
                content = f"""
                    Title: {title}
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
                logger.info(f"Document {i + doc_idx + 1}: Split into {len(chunks)} chunks")
                
                # Create metadata for each chunk
                for chunk in chunks:
                    doc_id = str(uuid.uuid4())
                    chunk_ids.append(doc_id)
                    all_chunks.append(chunk)
                    
                    # Create metadata with only the specified fields
                    metadata = prepare_safe_metadata(doc, doc_id, formatted_conf_name)
                    
                    # Add disease information if available (as a string to avoid metadata size issues)
                    if "disease" in doc and doc["disease"]:
                        # Store disease names as a string to avoid complex nested structures in metadata
                        metadata["disease_names"] = disease_info
                    
                    all_metadatas.append(metadata)
            
            # Generate embeddings for all chunks in this batch
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
            batch_embeddings = embedding_model.embed_documents(all_chunks)
            logger.info(f"Successfully generated {len(batch_embeddings)} embeddings")
            
            # Prepare vectors for upsert
            vectors = []
            for j, embedding in enumerate(batch_embeddings):
                vectors.append({
                    "id": chunk_ids[j],
                    "values": embedding,
                    "metadata": all_metadatas[j]
                })
            
            # Upsert batch of vectors
            logger.info(f"Upserting {len(vectors)} vectors to Pinecone index '{index_name}', namespace '{namespace}'")
            index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            logger.info(f"Successfully upserted vectors to Pinecone")
            
            # Add IDs to the global list
            document_ids.extend(chunk_ids)
            
            # Update job status
            processed_count += len(batch_docs)
            background_jobs[job_id]["processed_items"] = processed_count
            background_jobs[job_id]["status"] = f"Processed {processed_count} of {len(all_documents)} documents"
            
            # Log progress
            logger.info(f"Progress: {processed_count}/{len(all_documents)} documents processed ({(processed_count/len(all_documents)*100):.2f}%)")
        
        # Job completed successfully
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["document_ids"] = document_ids
        background_jobs[job_id]["count"] = len(document_ids)
        
        completion_message = f"Successfully processed {len(document_ids)} chunks from {len(all_documents)} documents"
        background_jobs[job_id]["message"] = completion_message
        logger.info(f"Job {job_id} completed: {completion_message}")
        
    except Exception as e:
        error_message = f"Job {job_id} failed: {str(e)}"
        logger.error(error_message, exc_info=True)
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["message"] = str(e)

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
    logger.info(f"Received request to create embeddings for {len(request.documents)} documents in index '{request.index_name}'")
    
    try:
        # Create index if it doesn't exist
        if not pc.has_index(request.index_name):
            logger.info(f"Creating new index: {request.index_name}")
            pc.create_index(
                name=request.index_name,
                dimension=request.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            logger.info(f"Index {request.index_name} already exists")
        
        # Get index
        index = pc.Index(request.index_name)
        
        # Process documents
        logger.info(f"Initializing text splitter with chunk_size={request.chunk_size}, chunk_overlap={request.chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size, 
            chunk_overlap=request.chunk_overlap
        )
        
        document_ids = []
        all_chunks = []
        all_metadatas = []
        
        for doc_idx, doc in enumerate(request.documents):
            # Format disease information
            disease_info = "N/A"
            if "disease" in doc and doc["disease"]:
                disease_names = [d.get("name", "") for d in doc["disease"] if d.get("name")]
                disease_info = ', '.join(disease_names)
            
            # Create content to be embedded
            title = doc.get('session_title', '') or doc.get('brief_title', '') or 'N/A'
            content = f"""
                Title: {title}
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
            logger.info(f"Document {doc_idx + 1}: Split into {len(chunks)} chunks")
            
            # Create metadata for each chunk
            for chunk in chunks:
                doc_id = str(uuid.uuid4())
                document_ids.append(doc_id)
                all_chunks.append(chunk)
                
                # Create metadata with only the specified fields
                metadata = prepare_safe_metadata(doc, doc_id)
                
                # Add disease information if available (as a string to avoid metadata size issues)
                if "disease" in doc and doc["disease"]:
                    metadata["disease_names"] = disease_info
                
                all_metadatas.append(metadata)
        
        # Initialize embedding model (only once)
        if request.embedding_type.lower() == "openai":
            logger.info("Initializing OpenAI embedding model (text-embedding-3-large)")
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-large", 
                openai_api_key=config.openai_api_key
            )
            
            # Process in batches
            batch_size = 100  # Adjust as needed
            total_chunks = len(all_chunks)
            logger.info(f"Processing {total_chunks} chunks in batches of {batch_size}")
            
            for i in range(0, total_chunks, batch_size):
                batch_end = min(i + batch_size, total_chunks)
                logger.info(f"Processing chunks {i} to {batch_end-1}")
                
                # Get batch of chunks and their IDs
                batch_chunks = all_chunks[i:batch_end]
                batch_ids = document_ids[i:batch_end]
                batch_metadatas = all_metadatas[i:batch_end]
                
                # Generate embeddings for the batch
                logger.info(f"Generating embeddings for batch of {len(batch_chunks)} chunks")
                batch_embeddings = embedding_model.embed_documents(batch_chunks)
                logger.info(f"Successfully generated {len(batch_embeddings)} embeddings")
                
                # Prepare vectors for upsert
                vectors = []
                for j, embedding in enumerate(batch_embeddings):
                    vectors.append({
                        "id": batch_ids[j],
                        "values": embedding,
                        "metadata": batch_metadatas[j]
                    })
                
                # Upsert batch of vectors
                logger.info(f"Upserting {len(vectors)} vectors to Pinecone index '{request.index_name}', namespace '{request.namespace}'")
                index.upsert(
                    vectors=vectors,
                    namespace=request.namespace
                )
                logger.info(f"Successfully upserted vectors to Pinecone")
        
        success_message = f"Successfully created embeddings for {len(document_ids)} chunks from {len(request.documents)} documents"
        logger.info(success_message)
        
        return EmbeddingResponse(
            success=True,
            message=success_message,
            document_ids=document_ids,
            count=len(document_ids)
        )
        
    except Exception as e:
        error_message = f"Error creating embeddings: {str(e)}"
        logger.error(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest_conference_data", response_model=EmbeddingResponse)
async def ingest_conference_data(
    request: ConferenceDataRequest,
    background_tasks: BackgroundTasks,
    pc: Pinecone = Depends(get_pinecone_client),
    config: Config = Depends(get_config)
):
    """
    Ingest conference data from the Larvol API and create embeddings.
    This will run as a background task and return a job ID for tracking progress.
    """
    logger.info(f"Received request to ingest conference data: {request.conferenceName} {request.conferenceIteration}")
    
    try:
        # Generate a job ID
        job_id = str(uuid.uuid4())
        logger.info(f"Generated job ID: {job_id}")
        namespace = f"{request.conferenceName}_{request.conferenceIteration}".lower().replace(" ", "_")
        logger.info(f"Using namespace: {namespace}")
        logger.info(f"Request details: {request}")
        # Initialize job status
        background_jobs[job_id] = {
            "status": "initializing",
            "processed_items": 0,
            "total_items": 0,
            "current_page": 0,
            "total_pages": 0,
            "namespace":namespace
        }
        
        # Start background task
        logger.info(f"Starting background task for job {job_id}")
        background_tasks.add_task(
            process_conference_data,
            request.conferenceName,
            request.conferenceIteration,
            request.index_name,
            namespace,
            request.chunk_size,
            request.chunk_overlap,
            request.max_pages,
            job_id,
            pc,
            config
        )
        
        success_message = f"Started ingestion of conference data for {request.conferenceName} {request.conferenceIteration}"
        logger.info(success_message)
        
        return EmbeddingResponse(
            success=True,
            message=success_message,
            job_id=job_id,
            count=0
        )
        
    except Exception as e:
        error_message = f"Error starting conference data ingestion: {str(e)}"
        logger.error(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/job_status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a background job.
    """
    logger.info(f"Received request for job status: {job_id}")
    
    if job_id not in background_jobs:
        logger.warning(f"Job ID not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = background_jobs[job_id]
    logger.info(f"Job status for {job_id}: {job.get('status', 'unknown')}")
    
    return JobStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        processed_items=job.get("processed_items", 0),
        total_items=job.get("total_items", 0),
        current_page=job.get("current_page", 0),
        total_pages=job.get("total_pages", 0),
        message=job.get("message")
    )
    
import os
import json
import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime

# Define your models
class QueryRequest(BaseModel):
    index_name: str = "conference-data"
    query_text: str
    top_k: int = 5
    embedding_type: str = "openai"
    namespace: Optional[str] = "default"
    include_metadata: bool = True
    filter: Optional[Dict[str, Any]] = None

class QueryResult(BaseModel):
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    results: List[QueryResult] = []
    count: int = 0
    query_time_ms: Optional[float] = None

class ConferenceQueryRequest(BaseModel):
    conferenceName: str
    conferenceIteration: str
    query: str
    top_k: int = 5
    generate_explanation: bool = True
    llm_model: str = "gemini-1.5-pro"  # or "gpt-4" or others

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]

class ConferenceQueryResponse(BaseModel):
    query: str
    vector_results: List[QueryResult]
    explanation: Optional[str] = None
    source_documents: List[SourceDocument] = []
    timing: Dict[str, float] = {}

# Add this to your existing FastAPI application
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
        logger.info(f"Processing query request for index: {request.index_name}, namespace: {request.namespace}")
        
        # Check if index exists
        if not pc.has_index(request.index_name):
            logger.error(f"Index {request.index_name} does not exist")
            raise HTTPException(status_code=404, detail=f"Index {request.index_name} does not exist")
        
        # Get index
        index = pc.Index(request.index_name)
        
        # Generate query embedding
        start_time = time.time()
        if request.embedding_type.lower() == "openai":
            logger.info("Generating OpenAI embedding for query")
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-large", 
                openai_api_key=config.openai_api_key
            )
            query_embedding = embedding_model.embed_query(request.query_text)
        elif request.embedding_type.lower() == "llama":
            logger.info("Generating Llama embedding for query")
            embedding_response = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[request.query_text],
                parameters={"input_type": "query", "truncate": "END", "dimension": 2048},
            )
            query_embedding = embedding_response[0]['values']
        else:
            logger.error(f"Unsupported embedding type: {request.embedding_type}")
            raise HTTPException(status_code=400, detail=f"Unsupported embedding type: {request.embedding_type}")
        
        embedding_time = time.time() - start_time
        logger.info(f"Embedding generation time: {embedding_time:.4f}s")
        
        # Query the index
        logger.info(f"Querying Pinecone index with top_k={request.top_k}")
        start_time = time.time()
        query_response = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=request.include_metadata,
            namespace=request.namespace,
            filter=request.filter
        )
        query_time = time.time() - start_time
        logger.info(f"Pinecone query time: {query_time:.4f}s")
        
        # Format results
        results = []
        for match in query_response.matches:
            results.append(QueryResult(
                id=match.id,
                score=match.score,
                metadata=match.metadata
            ))
        
        logger.info(f"Query returned {len(results)} results")
        
        return QueryResponse(
            results=results,
            count=len(results),
            query_time_ms=query_response.usage.total_ms if hasattr(query_response, 'usage') else query_time * 1000
        )
        
    except Exception as e:
        logger.error(f"Error in query_embeddings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conference_query", response_model=ConferenceQueryResponse)
async def query_conference_data(
    request: ConferenceQueryRequest,
    pc: Pinecone = Depends(get_pinecone_client),
    config: Config = Depends(get_config)
):
    """
    Query conference data with RAG capabilities to both retrieve vector results 
    and generate human-readable explanations.
    """
    start_time_total = time.time()
    timing = {}
    
    try:
        # Format namespace from conference name and iteration
        namespace = f"{request.conferenceName}_{request.conferenceIteration}".lower()
        index_name = "conference-data"  # Default index name
        
        logger.info(f"Processing conference query: '{request.query}' for {request.conferenceName} {request.conferenceIteration}")
        logger.info(f"Using index: {index_name}, namespace: {namespace}")
        
        # Check if index exists
        if not pc.has_index(index_name):
            logger.error(f"Index {index_name} does not exist")
            raise HTTPException(status_code=404, detail=f"Index {index_name} does not exist")
        
        # Get index
        index = pc.Index(index_name)
        
        # Initialize embedding model
        start_time = time.time()
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large", 
            openai_api_key=config.openai_api_key
        )
        embedding_init_time = time.time() - start_time
        timing["embedding_initialization"] = embedding_init_time
        logger.info(f"Embedding model initialized in {embedding_init_time:.4f}s")
        
        # First approach: Direct query to get vector results
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = embedding_model.embed_query(request.query)
        
        # Query the index
        query_response = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True,
            namespace=namespace,
        )
        
        vector_query_time = time.time() - start_time
        timing["vector_query_time"] = vector_query_time
        logger.info(f"Vector query completed in {vector_query_time:.4f}s")
        
        # Format vector results
        vector_results = []
        source_documents = []
        
        for match in query_response.matches:
            vector_results.append(QueryResult(
                id=match.id,
                score=match.score,
                metadata=match.metadata
            ))
            
            # Extract the content from metadata for source documents
            source_content = ""
            if match.metadata:
                # Try to construct meaningful content from metadata
                title = match.metadata.get("title", "No Title")
                category = match.metadata.get("category", "")
                date = match.metadata.get("date", "")
                
                # Some metadata contains full content in a field
                # But for our conference data, we need to reconstruct it
                source_content = f"Title: {title}\n"
                if date:
                    source_content += f"Date: {date}\n"
                if category:
                    source_content += f"Category: {category}\n"
                
                # Add any other relevant metadata fields
                for key, value in match.metadata.items():
                    if key not in ["title", "category", "date"] and value and isinstance(value, str):
                        source_content += f"{key.replace('_', ' ').title()}: {value}\n"
                        
            source_documents.append(SourceDocument(
                content=source_content,
                metadata=match.metadata or {}
            ))
        
        explanation = None
        
        # Second approach: Use LangChain for RAG if requested
        if request.generate_explanation:
            start_time = time.time()
            logger.info(f"Generating explanation using LLM model: {request.llm_model}")
            
            try:
                # Initialize vector store using LangChain
                vectorstore = PineconeVectorStore(
                    index=index,
                    embedding=embedding_model,
                    namespace=namespace
                )
                
                # Create retriever
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": request.top_k}
                )
                
                # Initialize LLM
                if request.llm_model.startswith("gemini"):
                    google_api_key = os.environ.get("GOOGLE_API_KEY")
                    if not google_api_key:
                        logger.error("GOOGLE_API_KEY not found in environment variables")
                        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")
                    
                    llm = ChatGoogleGenerativeAI(
                        model=request.llm_model,
                        google_api_key=google_api_key,
                        temperature=0
                    )
                else:
                    # Default to using other models if needed
                    logger.error(f"Unsupported LLM model: {request.llm_model}")
                    raise HTTPException(status_code=400, detail=f"Unsupported LLM model: {request.llm_model}")
                
                # Define prompt template
                prompt_template = """You are an expert medical orator which analyzes abstracts of the given medical conference and tries to answer the user question.
                If the abstracts contain ANY information related to the question, even if incomplete, summarize what is available, and always stick to the given abstract information. Do not provide any other information from your own knowledge.
                Only say you cannot answer if there is absolutely no relevant information, try to give answer in bullet points which is easier to understand and try to go deep in the abstract and identify all necessary information for the given query and summarize well.

                QUESTION: {question}

                RETRIEVED ABSTRACTS:
                {context}

                ANSWER:
                """

                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
                # Execute the query
                result = qa_chain({"query": request.query})
                explanation = result["result"]
                
                # Update source documents with more detailed information if available
                if "source_documents" in result:
                    source_documents = []
                    for doc in result["source_documents"]:
                        source_documents.append(SourceDocument(
                            content=doc.page_content,
                            metadata=doc.metadata
                        ))
                
                logger.info("LLM explanation generated successfully")
                
            except Exception as e:
                logger.error(f"Error generating explanation: {str(e)}", exc_info=True)
                explanation = f"Error generating explanation: {str(e)}"
            
            explanation_time = time.time() - start_time
            timing["explanation_generation_time"] = explanation_time
            logger.info(f"Explanation generated in {explanation_time:.4f}s")
        
        # Calculate total time
        total_time = time.time() - start_time_total
        timing["total_time"] = total_time
        
        return ConferenceQueryResponse(
            query=request.query,
            vector_results=vector_results,
            explanation=explanation,
            source_documents=source_documents,
            timing=timing
        )
        
    except Exception as e:
        logger.error(f"Error in conference query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))