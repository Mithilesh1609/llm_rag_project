from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
import uuid
import requests
import time
import logging
import fastapi
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

class CreateEmbeddingsRequest(BaseModel):
    index_name: str
    dimension: int = 3076
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
    chunk_size: int = 2000
    chunk_overlap: int = 300
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
            'Authorization': os.getenv("LARVOL_API_KEY")
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
        # all_documents_json = json.loads(all_documents)
        # print("type of all_documents: ",type(all_documents))
        # with open("all_documents.json", "w") as f:
        #     json.dump(all_documents, f)
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
        
        # Initialize skip_count to track records we want to skip
        skip_count = 0
        logger.info(f"Will skip processing first {skip_count} records")
        
        for i in range(0, len(all_documents), batch_size):
            batch_docs = all_documents[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(all_documents) + batch_size - 1)//batch_size} ({len(batch_docs)} documents)")
            
            all_chunks = []
            all_metadatas = []
            chunk_ids = []
            documents = []
            print("batch_doc len: ",len(batch_docs))
            # Calculate how many docs to skip in this batch
            docs_to_process = []
            for doc_idx, doc in enumerate(batch_docs):
                # Update the processed counter first
                processed_count += 1
                
                # Skip if we haven't reached the threshold yet
                if processed_count <= skip_count:
                    logger.info(f"Skipping document {processed_count} of {skip_count}")
                    continue
                    
                # Only process docs beyond our skip threshold
                docs_to_process.append((doc_idx, doc))
                
            # Update job status after counting
            background_jobs[job_id]["processed_items"] = processed_count
            background_jobs[job_id]["status"] = f"Processed {processed_count} of {len(all_documents)} documents, skipping first {skip_count}"
            
            # Skip further processing if no docs to process in this batch
            if not docs_to_process:
                logger.info(f"No documents to process in this batch after skipping threshold")
                continue
                
            logger.info(f"Processing {len(docs_to_process)} documents in this batch (after skipping threshold)")
            
            # Process only the docs we want to keep
            for doc_idx, doc in docs_to_process:
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
                logger.info(f"Document {processed_count - len(docs_to_process) + doc_idx + 1}: Split into {len(chunks)} chunks")
                
                # Create metadata for each chunk
                for chunk in chunks:
                    doc_id = str(uuid.uuid4())
                    chunk_ids.append(doc_id)
                    
                    # Create metadata with only the specified fields
                    metadata = prepare_safe_metadata(doc, doc_id, formatted_conf_name)
                    
                    # Add disease information if available (as a string to avoid metadata size issues)
                    if "disease" in doc and doc["disease"]:
                        # Store disease names as a string to avoid complex nested structures in metadata
                        metadata["disease_names"] = disease_info
                    documents.append(Document(page_content=chunk, metadata=metadata,id=doc_id))
            
            batch_docs_to_process = documents
            batch_texts_to_process = [doc.page_content for doc in batch_docs_to_process]
            logger.info(f"Generating embeddings for {len(batch_texts_to_process)} chunks")
            batch_embeddings = embedding_model.embed_documents(batch_texts_to_process)
            logger.info(f"Successfully generated {len(batch_embeddings)} embeddings")
            vectors = []

            for j, embedding_values in enumerate(batch_embeddings):
            # Include the page_content in the metadata
                metadata = batch_docs_to_process[j].metadata.copy()
                metadata["page_content"] = batch_docs_to_process[j].page_content
                vectors.append((str(i + j), embedding_values, metadata))
            logger.info(f"Upserting {len(vectors)} vectors to Pinecone index '{index_name}', namespace '{namespace}'")
            index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            logger.info(f"Successfully upserted vectors to Pinecone")
            
            # Add IDs to the global list
            document_ids.extend(chunk_ids)
            
            # Log progress - Note: processed_count is already updated earlier in the loop
            logger.info(f"Progress: {processed_count}/{len(all_documents)} documents seen, processing documents > {skip_count}")
            
            # Update job status with specific threshold information
            if processed_count <= skip_count:
                logger.info(f"Still below threshold of {skip_count}, no documents processed yet")
            else:
                logger.info(f"Processed documents {skip_count+1} to {processed_count} out of {len(all_documents)} total documents")
            
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
import requests
import time
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
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
    content: Optional[str] = None

class QueryResponse(BaseModel):
    results: List[QueryResult]
    count: int
    query_time_ms: Optional[float] = None
    
class ConferenceQueryRequest(BaseModel):
    conferenceName: str
    conferenceIteration: str
    query: str
    top_k: int = 5
    use_rag: bool = True
    model_name: str = "gpt-4o-mini"
    temperature: float = 0

class RagResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float

# Define prompt template for RAG
#### v1 till may-12-2025
# prompt_template = """You are an expert medical orator who analyzes abstracts from medical conferences to answer user questions.
# If the abstracts contain ANY information related to the question, even if incomplete, summarize what is available, and always stick to the given abstracts. Do not provide any information beyond what's in the abstracts.
# Only say you cannot answer if there is absolutely no relevant information. Present your answer in bullet points for easier understanding.
# Try to go deep into the abstracts and identify all necessary information for the given query and summarize well.

# QUESTION: {question}

# RETRIEVED ABSTRACTS:
# {context}

# ANSWER:
# """

prompt_template = """You are an expert medical orator who analyzes abstracts from medical conferences to answer user questions.
If the abstracts contain ANY information related to the question, even if incomplete, summarize what is available, and always stick to the given abstracts. Do not provide any information beyond what's in the abstracts.
Only say you cannot answer if there is absolutely no relevant information. Present your answer in bullet points for easier understanding.
Try to go deep into the abstracts and identify all necessary information for the given query and summarize well, please try to more analytical and to the point as you are givinig information to medical people who will be in hurry and required to get information in a concise and efficient manner, and highlight key aspect of the answer by bolding it and follow the markdown format for efficient formatting, and stick to given data only, do not add details from other sources.
# And Try to sound like a helper which has good technical know how but also great at communicating complex stuff in easier way for people.
QUESTION: {question}

RETRIEVED ABSTRACTS:
{context}

ANSWER:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Query API without-streaming
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
        logger.info(f"Processing query: '{request.query_text}' on index '{request.index_name}'")
        
        # Check if index exists
        if not pc.has_index(request.index_name):
            logger.error(f"Index {request.index_name} does not exist")
            raise HTTPException(status_code=404, detail=f"Index {request.index_name} does not exist")
        
        # Get index
        index = pc.Index(request.index_name)
        
        # Generate query embedding
        start_time = time.time()
        if request.embedding_type.lower() == "openai":
            logger.info("Generating embedding using OpenAI model")
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-large", 
                openai_api_key=config.openai_api_key
            )
            query_embedding = embedding_model.embed_query(request.query_text)
            
        elif request.embedding_type.lower() == "llama":
            logger.info("Generating embedding using Llama model")
            embedding_response = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[request.query_text],
                parameters={"input_type": "query", "truncate": "END", "dimension": 2048},
            )
            query_embedding = embedding_response[0]['values']
        
        embedding_time = time.time() - start_time
        logger.info(f"Embedding generation completed in {embedding_time:.4f}s")
        
        # Query the index
        logger.info(f"Querying Pinecone index with top_k={request.top_k}")
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
        
        logger.info(f"Query completed with {len(results)} results")
        
        return QueryResponse(
            results=results,
            count=len(results),
            query_time_ms=query_response.usage.total_ms if hasattr(query_response, 'usage') else None
        )
        
    except Exception as e:
        logger.error(f"Error in query_embeddings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class DeleteEmbeddingsRequest(BaseModel):
    index_name: str
    ids: List[str]
    namespace: Optional[str] = "default"
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

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class StreamingCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tokens = []
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)
        self.text += token

@app.post("/api/conference/query_streaming")
async def query_conference_data(
    request: ConferenceQueryRequest,
    response: fastapi.Response,
    background_tasks: BackgroundTasks,
    pc: Pinecone = Depends(get_pinecone_client),
    config: Config = Depends(get_config)
):
    """
    Query conference data and optionally use RAG to generate a natural language response
    with simplified streaming capability for real-time response display.
    """
    try:
        logger.info(f"Processing conference query: '{request.query}' for {request.conferenceName} {request.conferenceIteration}")
        
        # Create namespace from conference name and iteration
        namespace = f"{request.conferenceName.lower()}_{request.conferenceIteration}"
        index_name = "conference-data"
        
        # Check if index exists
        if not pc.has_index(index_name):
            logger.error(f"Index {index_name} does not exist")
            raise HTTPException(status_code=404, detail=f"Index {index_name} does not exist")
        
        # Get index
        index = pc.Index(index_name)
        
        # Track timing
        start_time_total = time.time()
        retrieval_start_time = time.time()
        
        # Initialize embedding model
        logger.info("Initializing OpenAI embedding model")
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large", 
            openai_api_key=config.openai_api_key
        )
        
        # If using RAG, set up the retrieval system
        if request.use_rag:
            logger.info("Setting up RAG system with LangChain")
            
            # Create vector store
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=embedding_model,
                text_key="page_content"
            )
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={
                    "k": request.top_k,
                    "namespace": namespace
                }
            )
            
            # Retrieve relevant documents first
            logger.info("Retrieving relevant documents")
            relevant_docs = retriever.get_relevant_documents(request.query)
            
            # Record retrieval time
            retrieval_time = time.time() - retrieval_start_time
            logger.info(f"Retrieved {len(relevant_docs)} documents in {retrieval_time:.4f}s")
            
            # Format sources for the response
            sources = []
            for doc in relevant_docs:
                source_data = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_data)
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Initialize LLM with streaming capability
            logger.info(f"Initializing LLM with streaming: {request.model_name}")
            llm = ChatOpenAI(
                model_name=request.model_name, 
                openai_api_key=config.openai_api_key, 
                temperature=request.temperature,
                streaming=True
            )
            
            # Prepare the prompt
            formatted_prompt = PROMPT.format(
                context=context,
                question=request.query
            )
            
            # Set up streaming response
            response.headers["Content-Type"] = "text/plain"
            response.headers["Cache-Control"] = "no-cache"
            response.headers["Connection"] = "keep-alive"
            
            accumulated_answer = ""
            generation_start_time = time.time()
            
            async def generate_stream():
                nonlocal accumulated_answer
                
                # Stream the response content directly as plain text
                async for chunk in llm.astream(formatted_prompt):
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    accumulated_answer += content
                    yield content
                
                # Calculate final metrics (these won't be seen by client in streaming mode)
                generation_time = time.time() - generation_start_time
                total_time = time.time() - start_time_total
                
                logger.info(f"RAG query completed in {total_time:.4f}s (retrieval: {retrieval_time:.4f}s, generation: {generation_time:.4f}s)")
                
                # Store the complete response data for later retrieval if needed
                response_data = {
                    "query": request.query,
                    "answer": accumulated_answer,
                    "sources": sources,
                    "retrieval_time_ms": retrieval_time * 1000,
                    "generation_time_ms": generation_time * 1000,
                    "total_time_ms": total_time * 1000
                }
                
                # Store in a cache or database if needed for later retrieval
                # background_tasks.add_task(store_response_data, response_data)
            
            # Return the streaming generator with plain text content
            return fastapi.responses.StreamingResponse(
                generate_stream(),
                media_type="text/plain"
            )
            
        else:
            # Perform direct vector search without RAG
            logger.info("Performing direct vector search without RAG")
            
            # Generate query embedding
            query_embedding = embedding_model.embed_query(request.query)
            
            # Query the index
            query_response = index.query(
                vector=query_embedding,
                top_k=request.top_k,
                include_metadata=True,
                namespace=namespace
            )
            
            # Format sources
            sources = []
            for match in query_response.matches:
                content = match.metadata.get("page_content", "Content not available")
                source_data = {
                    "content": content,
                    "metadata": match.metadata,
                    "score": match.score
                }
                sources.append(source_data)
            
            # Calculate times
            retrieval_time = time.time() - retrieval_start_time
            total_time = time.time() - start_time_total
            
            logger.info(f"Vector search completed in {total_time:.4f}s")
            
            # For non-RAG searches, return a JSON response
            return RagResponse(
                query=request.query,
                answer="Vector search results only (no RAG generation was requested)",
                sources=sources,
                retrieval_time_ms=retrieval_time * 1000,
                generation_time_ms=0,
                total_time_ms=total_time * 1000
            )
        
    except Exception as e:
        logger.error(f"Error in query_conference_data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
      
######## update functionality

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    processed_items: int = 0
    total_items: int = 0
    updated_items: int = 0
    skipped_items: int = 0
    current_page: int = 0
    total_pages: int = 0
    message: Optional[str] = None
async def update_conference_embeddings(
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
    """Background task to update conference data embeddings only if version has changed"""
    try:
        logger.info(f"Starting update job {job_id} for conference {conference_name} {conference_iteration}")
        
        background_jobs[job_id] = {
            "status": "running", 
            "processed_items": 0, 
            "total_items": 0,
            "updated_items": 0,
            "skipped_items": 0,
            "current_page": 0,
            "total_pages": 0
        }
        
        # Format the conference name for the API
        formatted_conf_name = f"{conference_name} {conference_iteration}"
        logger.info(f"Formatted conference name: {formatted_conf_name}")
        
        # Fetch first page to get total pages
        logger.info(f"Fetching first page to determine total pages")
        documents, total_pages = fetch_conference_data(formatted_conf_name, 1)
        
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
                dimension=3072,  # Using dimension=3072 for text-embedding-3-large
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
        updated_count = 0
        skipped_count = 0
        
        # Process in batches
        batch_size = 20  # Adjust based on API rate limits and memory constraints
        logger.info(f"Processing documents in batches of {batch_size}")
        
        for i in range(0, len(all_documents), batch_size):
            batch_docs = all_documents[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(all_documents) + batch_size - 1)//batch_size} ({len(batch_docs)} documents)")
            
            # Process each document in the batch
            for doc_idx, doc in enumerate(batch_docs):
                source_id = str(doc.get("source_id", ""))
                if not source_id:
                    logger.warning(f"Document at index {i + doc_idx} has no source_id, skipping")
                    continue
                
                # Check if document already exists in the index and get its metadata
                # Query by source_id to find existing records
                existing_docs = index.query(
                    vector=[0] * 3072,  # Dummy vector for metadata-only query
                    namespace=namespace,
                    filter={"source_id": source_id},
                    include_metadata=True,
                    top_k=1
                )
                
                # Get the new Conf_Upload_Version
                new_version = str(doc.get("Conf_Upload_Version", ""))
                
                # Check if we need to update this document
                update_needed = True
                if existing_docs and existing_docs.matches:
                    existing_metadata = existing_docs.matches[0].metadata
                    existing_version = existing_metadata.get("Conf_Upload_Version", "")
                    
                    # If versions match, no need to update
                    if existing_version == new_version and new_version:
                        logger.info(f"Document with source_id {source_id} already has version {new_version}, skipping")
                        skipped_count += 1
                        continue
                
                # If we reached here, we need to update or create this document
                logger.info(f"Updating/creating document with source_id {source_id}, version: {new_version}")
                
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
                logger.info(f"Document {i + doc_idx + 1}: Split into {len(chunks)} chunks")
                
                # Delete any existing documents with this source_id
                if existing_docs and existing_docs.matches:
                    try:
                        # Get all existing documents with this source_id to delete them
                        all_existing = index.query(
                            vector=[0] * 3072,
                            namespace=namespace,
                            filter={"source_id": source_id},
                            include_metadata=True,
                            top_k=100  # Assuming no more than 100 chunks per doc
                        )
                        
                        ids_to_delete = [m.id for m in all_existing.matches]
                        if ids_to_delete:
                            logger.info(f"Deleting {len(ids_to_delete)} existing chunks for source_id {source_id}")
                            index.delete(ids=ids_to_delete, namespace=namespace)
                    except Exception as e:
                        logger.error(f"Error deleting existing documents: {str(e)}")
                
                # Process and create new chunks
                vectors = []
                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = str(uuid.uuid4())
                    document_ids.append(doc_id)
                    
                    # Create metadata
                    metadata = prepare_safe_metadata(doc, doc_id, formatted_conf_name)
                    
                    # Ensure Conf_Upload_Version is included in metadata
                    if new_version:
                        metadata["Conf_Upload_Version"] = new_version
                    
                    # Create embedding for this chunk
                    embedding = embedding_model.embed_query(chunk)
                    
                    # Add content to metadata
                    metadata["page_content"] = chunk
                    
                    # Add to vectors for batch upsert
                    vectors.append((doc_id, embedding, metadata))
                
                # Upsert vectors if we have any
                if vectors:
                    logger.info(f"Upserting {len(vectors)} new vectors for source_id {source_id}")
                    index.upsert(vectors=vectors, namespace=namespace)
                    updated_count += 1
                
                processed_count += 1
            
            # Update job status
            background_jobs[job_id]["processed_items"] = processed_count
            background_jobs[job_id]["updated_items"] = updated_count
            background_jobs[job_id]["skipped_items"] = skipped_count
            background_jobs[job_id]["status"] = f"Processed {processed_count} of {len(all_documents)} documents (Updated: {updated_count}, Skipped: {skipped_count})"
            
            # Log progress
            logger.info(f"Progress: {processed_count}/{len(all_documents)} documents processed ({(processed_count/len(all_documents)*100):.2f}%)")
        
        # Job completed successfully
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["document_ids"] = document_ids
        background_jobs[job_id]["count"] = len(document_ids)
        
        completion_message = f"Successfully processed {len(document_ids)} chunks from {len(all_documents)} documents. Updated {updated_count}, skipped {skipped_count}."
        background_jobs[job_id]["message"] = completion_message
        logger.info(f"Job {job_id} completed: {completion_message}")
        
    except Exception as e:
        error_message = f"Job {job_id} failed: {str(e)}"
        logger.error(error_message, exc_info=True)
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["message"] = str(e)

@app.post("/api/update_conference_data", response_model=EmbeddingResponse)
async def update_conference_data(
    request: ConferenceDataRequest,
    background_tasks: BackgroundTasks,
    pc: Pinecone = Depends(get_pinecone_client),
    config: Config = Depends(get_config)
):
    """
    Update conference data embeddings, but only if the Conf_Upload_Version has changed.
    This will run as a background task and return a job ID for tracking progress.
    """
    logger.info(f"Received request to update conference data: {request.conferenceName} {request.conferenceIteration}")
    
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
            "updated_items": 0,
            "skipped_items": 0,
            "current_page": 0,
            "total_pages": 0,
            "namespace": namespace
        }
        
        # Start background task
        logger.info(f"Starting background task for job {job_id}")
        background_tasks.add_task(
            update_conference_embeddings,
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
        
        success_message = f"Started updating conference data for {request.conferenceName} {request.conferenceIteration}"
        logger.info(success_message)
        
        return EmbeddingResponse(
            success=True,
            message=success_message,
            job_id=job_id,
            count=0
        )
        
    except Exception as e:
        error_message = f"Error starting conference data update: {str(e)}"
        logger.error(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))