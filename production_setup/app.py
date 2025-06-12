from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

from datetime import datetime
import os, re
import uuid
import requests
import time
import logging
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import fastapi
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Header, status
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
import threading
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Thread-safe stats tracking"""
    def __init__(self):
        self.processed_items = 0
        self.total_items = 0
        self.current_page = 0
        self.total_pages = 0
        self.failed_pages = 0
        self._lock = threading.Lock()
    
    def update_processed(self, count: int):
        with self._lock:
            self.processed_items += count
    
    def set_totals(self, total_items: int, total_pages: int):
        with self._lock:
            self.total_items = total_items
            self.total_pages = total_pages
    
    def increment_failed_pages(self):
        with self._lock:
            self.failed_pages += 1
    
    def get_stats(self):
        with self._lock:
            return {
                "processed_items": self.processed_items,
                "total_items": self.total_items,
                "current_page": self.current_page,
                "total_pages": self.total_pages,
                "failed_pages": self.failed_pages
            }

def extract_acronym_data(input_string):
    """
    Extracts the last four digits as 'year' and converts the rest into
    'ONEWORD-SECONDWORD' uppercase format.
    """
    year_pattern = r'(\d{4})$'
    year_match = re.search(year_pattern, input_string)

    if not year_match:
        return None, None

    year = year_match.group(1)
    rest_of_string = input_string[:year_match.start()]

    cleaned_parts = re.split(r'[-_ ]+', rest_of_string)
    cleaned_parts = [part.strip() for part in cleaned_parts if part.strip()]

    formatted_rest = ""
    if len(cleaned_parts) >= 2:
        formatted_rest = f"{cleaned_parts[0].upper()}-{cleaned_parts[1].upper()}"
    elif len(cleaned_parts) == 1:
        formatted_rest = cleaned_parts[0].upper()

    return formatted_rest, year

app = FastAPI(title="Vector Database API", 
              description="API for managing document embeddings in Pinecone")

API_KEY = os.environ.get("X_API_KEY")

async def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    """Verifies the x-api-key provided in the request header."""
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid X-API-Key",
        )
    return x_api_key

class Config:
    def __init__(self):
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.larvol_api_key = os.environ.get("LARVOL_API_KEY")
        
        if not self.pinecone_api_key:
            logger.error("PINECONE_API_KEY environment variable not set")
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            
        if not self.pinecone_api_key or not self.openai_api_key:
            raise ValueError("Missing required API keys in environment variables")
        
        logger.info("Configuration initialized successfully")

def get_config():
    return Config()

def get_pinecone_client(config: Config = Depends(get_config)):
    logger.info("Initializing Pinecone client")
    return Pinecone(api_key=config.pinecone_api_key)

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
    chunk_size: int = 400
    chunk_overlap: int = 100
    embedding_type: str = "openai"
    documents: List[Dict[str, Any]]

class EmbeddingResponse(BaseModel):
    success: bool
    message: str
    document_ids: List[str] = []
    count: int = 0
    job_id: Optional[str] = None

class ConferenceDataRequest(BaseModel):
    conferenceAcronym: str
    conferenceName: str = ""
    conferenceIteration: str = ""
    timeStamp: datetime = ""
    index_name: str = "conference-data-production"
    namespace: str = "default"
    chunk_size: int = 400
    chunk_overlap: int = 100
    max_pages: int = 120
    max_parallel_requests: int = 5  # Control parallel API calls
    batch_size: int = 10  # Smaller batch sizes for Pinecone upsert
    max_batch_size_mb: float = 1.9  # Maximum batch size in MB (keeping under 2MB limit)

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    processed_items: int = 0
    total_items: int = 0
    current_page: int = 0
    total_pages: int = 0
    failed_pages: int = 0
    message: Optional[str] = None

# Dictionary to track background jobs
background_jobs = {}

def fetch_conference_data_page(conference_name: str, page: int, timestamp: datetime = "", max_retries: int = 3):
    """Fetch conference data from the API for a specific page with retry logic"""
    
    if timestamp != "":
        api_url = f"https://lt.larvol.com/api/news.php?paged_mode=yes&page={page}&conf_name={conference_name}&timestamp={timestamp}"
    else:
        api_url = f"https://lt.larvol.com/api/news.php?paged_mode=yes&page={page}&conf_name={conference_name}"
    
    api_key = os.getenv("LARVOL_API_KEY")
    
    if not api_key:
        raise ValueError("LARVOL_API_KEY environment variable is not set")
    
    headers = {'Authorization': api_key}
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching page {page} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(api_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list) and len(data) >= 2:
                    total_pages = 1
                    if "total_pages" in data[0]:
                        try:
                            total_pages = int(data[0]["total_pages"])
                        except (ValueError, TypeError) as e:
                            logger.error(f"Error parsing total_pages: {e}")
                    
                    return data[1:], total_pages
                else:
                    if isinstance(data, dict) and "result" in data and data["result"] == "error":
                        error_msg = data.get("message", "Unknown API error")
                        raise ValueError(f"API error: {error_msg}")
                    
            else:
                logger.warning(f"Page {page} failed with status code: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise ValueError(f"Failed to fetch page {page} after {max_retries} attempts")
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for page {page}, attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise ValueError(f"Network error fetching page {page}: {e}")
    
    return [], 1

def fetch_conference_data_parallel(
    conference_name: str, 
    total_pages: int, 
    timestamp: datetime = "", 
    max_parallel_requests: int = 5,
    stats: ProcessingStats = None
):
    """Fetch conference data from multiple pages in parallel"""
    
    all_documents = []
    failed_pages = []
    
    def fetch_page_wrapper(page_num):
        try:
            documents, _ = fetch_conference_data_page(conference_name, page_num, timestamp)
            logger.info(f"Successfully fetched page {page_num}: {len(documents)} documents")
            return page_num, documents, None
        except Exception as e:
            error_msg = f"Failed to fetch page {page_num}: {str(e)}"
            logger.error(error_msg)
            if stats:
                stats.increment_failed_pages()
            return page_num, [], error_msg
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
        # Submit all page requests
        future_to_page = {
            executor.submit(fetch_page_wrapper, page): page 
            for page in range(1, total_pages + 1)
        }
        
        # Collect results as they complete
        page_results = {}
        for future in as_completed(future_to_page):
            page_num, documents, error = future.result()
            page_results[page_num] = (documents, error)
            
            if error:
                failed_pages.append(page_num)
            
            logger.info(f"Completed page {page_num} ({len(page_results)}/{total_pages})")
    
    # Combine results in page order
    for page_num in sorted(page_results.keys()):
        documents, error = page_results[page_num]
        if not error:
            all_documents.extend(documents)
    
    logger.info(f"Parallel fetch completed: {len(all_documents)} total documents, {len(failed_pages)} failed pages")
    
    if failed_pages:
        logger.warning(f"Failed to fetch pages: {failed_pages}")
    
    return all_documents, failed_pages

def estimate_batch_size_mb(vectors: List[tuple]) -> float:
    """Estimate the size of a batch in MB"""
    if not vectors:
        return 0.0
    
    # Rough estimation based on vector components
    total_size = 0
    for vector_id, embedding, metadata in vectors:
        # Vector ID size
        total_size += len(str(vector_id))
        
        # Embedding size (assuming float32)
        total_size += len(embedding) * 4
        
        # Metadata size (rough estimation)
        metadata_str = json.dumps(metadata, default=str)
        total_size += len(metadata_str.encode('utf-8'))
    
    return total_size / (1024 * 1024)  # Convert to MB

def prepare_safe_metadata(doc, doc_id, conference_name=None, max_string_length=1000):
    """
    Prepare metadata that is safe for Pinecone with size limits.
    """
    def truncate_string(value, max_length=max_string_length):
        """Truncate string to max length"""
        if not isinstance(value, str):
            value = str(value)
        return value[:max_length] if len(value) > max_length else value
    
    # Format disease information
    disease_info = ""
    if "disease" in doc and doc["disease"] and isinstance(doc["disease"], list):
        disease_names = [d.get("name", "") for d in doc["disease"] if d.get("name")]
        disease_info = ', '.join(disease_names)
    
    title = doc.get('session_title', '') or doc.get('brief_title', '') or ''
    
    # Create base metadata with truncated strings
    # metadata = {
    #     "title": truncate_string(title),
    #     "source_id": truncate_string(str(doc.get("source_id", ""))),
    #     "session_id": truncate_string(str(doc.get("session_id", ""))),
    #     "abstract_number": truncate_string(str(doc.get("abstract_number", ""))),
    #     "date": truncate_string(str(doc.get("date", ""))),
    #     "start_time": truncate_string(str(doc.get("start_time", ""))),
    #     "end_time": truncate_string(str(doc.get("end_time", ""))),
    #     "location": truncate_string(str(doc.get("location", ""))),
    #     "news_type": truncate_string(str(doc.get("news_type", ""))),
    #     "category": truncate_string(str(doc.get("category", ""))),
    #     "sub_category": truncate_string(str(doc.get("sub_category", ""))),
        
    #     "document_id": doc_id,
    # }
    
    metadata = {
        "sessionTitle":truncate_string(title),
        "conferenceDay":str(doc.get('day', 'N/A')),
        "conferenceDate":str(doc.get('date', 'N/A')),
        "phase":str(doc.get('phase', 'N/A')),
        "score":str(doc.get('score', 'N/A')),
        "source":str(doc.get('source', 'N/A')),
        "status":str(doc.get('status', 'N/A')),
        "abstractAuthors":str(doc.get('authors', 'N/A')),
        "discussedDisease":str(doc.get('disease', 'N/A')),
        "is_paid":str(doc.get('is_paid', 'N/A')),
        "sponser":str(doc.get('sponser', 'N/A')),
        "end_time":str(doc.get('end_time', 'N/A')),
        "start_time":str(doc.get('start_time', 'N/A')),
        "location":str(doc.get('location', 'N/A')),
        "biomarker":str(doc.get('biomarker', 'N/A')),
        "duplicate":str(doc.get('duplicate', 'N/A')),
        "news_type":str(doc.get('news_type', 'N/A')),
        "paperPresenters":str(doc.get('presenter', 'N/A')),
        "redtag_id":str(doc.get('redtag_id', 'N/A')),
        "abstract_id":str(doc.get('abstract_number', 'N/A')),
        "source_id":str(doc.get('source_id', 'N/A')),
        "trail_ids":str(doc.get('trail_ids', 'N/A')),
        "session_id":str(doc.get('session_id', 'N/A')),
        "institution":str(doc.get('institution', 'N/A')),
        "source_type":str(doc.get('source_type', 'N/A')),
        "investigator":str(doc.get('investigator', 'N/A')),
        "journal_name":str(doc.get('journal_name', 'N/A')),
        "session_type":str(doc.get('session_type', 'N/A')),
        "abstract_category":str(doc.get('category', 'N/A')),
        "abstract_sub_category":str(doc.get('sub_category', 'N/A')),
        "session_title":str(doc.get('session_title', 'N/A')),
        "generation_date":str(doc.get('generation_date', 'N/A')),
        "conference_source":str(doc.get('conference_source', 'N/A')),
        "conf_upload_version":str(doc.get('conf_upload_version', 'N/A')),
        "last_changed_date":str(doc.get('last_changed_date', 'N/A'))
    }
    
    
    # Add conference name if provided
    if conference_name:
        metadata["conference_name"] = truncate_string(str(conference_name))
    
    # Add disease information if available
    if disease_info:
        metadata["disease_names"] = truncate_string(disease_info)
    
    # Handle trial_ids specifically
    # if doc.get("trial_ids") is not None and doc.get("trial_ids") != "null":
    # metadata["trial_ids"] = truncate_string(str(doc.get("trial_ids")))
    
    # Add other fields with truncation
    # optional_fields = [
    #     "authors", "sponsor", "affiliations", 
    #     "Conf_Upload_Version", "session_text"
    # ]
    
    # for field in optional_fields:
    #     if field in doc and doc[field] is not None:
    #         # Use shorter truncation for potentially large fields
    #         # max_len = 500 if field in ["details", "summary", "session_text"] else max_string_length
    #         max_len = 500
    #         metadata[field] = truncate_string(str(doc[field]), max_len)
    
    return metadata

def create_optimized_batches(documents: List[Document], max_batch_size_mb: float = 1.5):
    """
    Create batches that respect the size limit for Pinecone upserts.
    """
    batches = []
    current_batch = []
    current_size_mb = 0.0
    
    for doc in documents:
        # Estimate size of this document when converted to vector format
        # This is a rough estimation
        doc_size_mb = len(doc.page_content.encode('utf-8')) / (1024 * 1024)
        doc_size_mb += len(json.dumps(doc.metadata, default=str).encode('utf-8')) / (1024 * 1024)
        doc_size_mb += 3072 * 4 / (1024 * 1024)  # Embedding size in MB
        
        if current_batch and (current_size_mb + doc_size_mb) > max_batch_size_mb:
            # Start new batch
            batches.append(current_batch)
            current_batch = [doc]
            current_size_mb = doc_size_mb
        else:
            current_batch.append(doc)
            current_size_mb += doc_size_mb
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

async def process_conference_data(
    conference_name: str,
    conference_iteration: str,
    index_name: str,
    namespace: str,
    chunk_size: int,
    chunk_overlap: int,
    max_pages: int,
    max_parallel_requests: int,
    batch_size: int,
    max_batch_size_mb: float,
    job_id: str,
    pc: Pinecone,
    config: Config,
    timestamp: datetime = ""
):
    """Background task to fetch conference data and create embeddings with parallel processing"""
    stats = ProcessingStats()
    
    try:
        logger.info(f"Starting job {job_id} for conference {conference_name} {conference_iteration}")
        
        background_jobs[job_id] = {
            "status": "running", 
            "processed_items": 0, 
            "total_items": 0,
            "current_page": 0,
            "total_pages": 0,
            "failed_pages": 0
        }
        
        # Format the conference name for the API
        formatted_conf_name = f"{conference_name} {conference_iteration}"
        logger.info(f"Formatted conference name: {formatted_conf_name}")
        
        # Fetch first page to get total pages
        logger.info(f"Fetching first page to determine total pages")
        _, total_pages = fetch_conference_data_page(formatted_conf_name, 1, timestamp)
        
        # Limit total pages to max_pages
        total_pages = min(total_pages, max_pages)
        logger.info(f"Will process {total_pages} pages (limited by max_pages={max_pages})")
        
        stats.set_totals(0, total_pages)  # Will update total_items later
        background_jobs[job_id]["total_pages"] = total_pages
        background_jobs[job_id]["status"] = f"Fetching data from {total_pages} pages in parallel"
        
        # Fetch all pages in parallel
        logger.info(f"Starting parallel fetch of {total_pages} pages with {max_parallel_requests} concurrent requests")
        all_documents, failed_pages = fetch_conference_data_parallel(
            formatted_conf_name, 
            total_pages, 
            timestamp, 
            max_parallel_requests,
            stats
        )
        
        # Update job status with fetch results
        total_document_count = len(all_documents)
        stats.set_totals(total_document_count, total_pages)
        
        background_jobs[job_id].update({
            "total_items": total_document_count,
            "failed_pages": len(failed_pages),
            "status": f"Processing {total_document_count} documents"
        })
        
        logger.info(f"Total documents collected: {total_document_count}")
        
        if failed_pages:
            logger.warning(f"Failed to fetch {len(failed_pages)} pages: {failed_pages}")
        
        # Create index if it doesn't exist
        logger.info(f"Checking if index {index_name} exists")
        if not pc.has_index(index_name):
            logger.info(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            time.sleep(10)
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
        
        # Process documents in batches
        logger.info(f"Processing documents in optimized batches (max {max_batch_size_mb}MB per batch)")
        
        # Convert documents to Document objects first
        all_doc_objects = []
        
        for doc_idx, doc in enumerate(all_documents):
            # Format disease information
            disease_info = "N/A"
            if "disease" in doc and doc["disease"]:
                disease_names = [d.get("name", "") for d in doc["disease"] if d.get("name")]
                disease_info = ', '.join(disease_names)
            
            # Create content to be embedded
            title = doc.get('session_title', '') or doc.get('brief_title', '') or 'N/A'
            
            # content = f"""
            #     Title: {title}
            #     Date: {doc.get('date', 'N/A')}
            #     Start Time: {doc.get('start_time', 'N/A')} 
            #     End Time: {doc.get('end_time', 'N/A')}
            #     Location: {doc.get('location', 'N/A')}
            #     Category: {doc.get('category', 'N/A')}
            #     Sub Category: {doc.get('sub_category', 'N/A')}
            #     Diseases: {disease_info}
            #     Sponsor: {doc.get('sponsor', 'N/A')}
            #     Session Text: {doc.get('session_text', 'N/A')}
            #     Disclosure: {doc.get('disclosures', 'N/A')}
            #     News Type: {doc.get('news_type', 'N/A')}
            #     Affiliation: {doc.get('affiliations', 'N/A')}
            #     Details: {doc.get('details', 'N/A')}
            #     Summary: {doc.get('summary', 'N/A')}
            #     Authors: {doc.get('authors', 'N/A')}
            #     Abstract Number: {doc.get('abstract_number', 'N/A')}
            #     Source Type: {doc.get('source_type', 'N/A')}
            #     Session Type: {doc.get('session_type', 'N/A')}
            #     Institution: {doc.get('institution', 'N/A')}
            #     Brief Title: {doc.get('brief_title', 'N/A')}
            #     Journal Name: {doc.get('journal_name', 'N/A')}
            #     Investigator: {doc.get('investigator', 'N/A')}
            #     KOL: {doc.get('kol', 'N/A')}
            # """.strip()
            content = f"""
                "sessionTitle": {title},
                "conferenceDay":{doc.get('day', 'N/A')},
                "conferenceDate":{doc.get('date', 'N/A')},
                "phase":{doc.get('phase', 'N/A')},
                "score":{doc.get('score', 'N/A')},
                "source":{doc.get('source', 'N/A')},
                "status":{doc.get('status', 'N/A')},
                "abstractAuthors":{doc.get('authors', 'N/A')},
                "paperAbstractText":{doc.get('details', 'N/A')},
                "discussedDisease":{doc.get('disease', 'N/A')},
                "is_paid":{doc.get('is_paid', 'N/A')},
                "sponsor":{doc.get('sponsor', 'N/A')},
                "abstractSummary":{doc.get('summary', 'N/A')},
                "abstract_category":{doc.get('category', 'N/A')},
                "abstract_sub_category":{doc.get('sub_category', 'N/A')},
                "end_time":{doc.get('end_time', 'N/A')},
                "start_time":{doc.get('start_time', 'N/A')},
                "location":{doc.get('location', 'N/A')},
                "biomarker":{doc.get('biomarker', 'N/A')},
                "duplicate":{doc.get('duplicate', 'N/A')},
                "news_type":{doc.get('news_type', 'N/A')},
                "paperPresenters":{doc.get('presenter', 'N/A')},
                "redtag_id":{doc.get('redtag_id', 'N/A')},
                "abstract_id":{doc.get('abstract_number', 'N/A')},
                "trial_ids":{doc.get('trial_ids', 'N/A')},                
                "session_id":{doc.get('session_id', 'N/A')},
                "alteration":{doc.get('alteration', 'N/A')},
                "enrollment":{doc.get('enrollment', 'N/A')},
                "disclosures":{doc.get('disclosures', 'N/A')},
                "institution":{doc.get('institution', 'N/A')},
                "primaryDrugProduct":{doc.get('primary_moa', 'N/A')},
                "secondaryDrugProduct":{doc.get('secondary_moa', 'N/A')},
                "source_type":{doc.get('source_type', 'N/A')},
                "abstract_collaborators":{doc.get('affiliations', 'N/A')},
                "investigator":{doc.get('investigator', 'N/A')},
                "journal_name":{doc.get('journal_name', 'N/A')},
                "poster_board":{doc.get('poster_board', 'N/A')},
                "session_text":{doc.get('session_text', 'N/A')},
                "session_type":{doc.get('session_type', 'N/A')},
                "session_title":{doc.get('session_title', 'N/A')},
                "upstream_areas":{doc.get('upstream_areas', 'N/A')},
                "generation_date":{doc.get('generation_date', 'N/A')},
                "primary_product":{doc.get('primary_product', 'N/A')},
                "conference_source":{doc.get('conference_source', 'N/A')},
                "conf_upload_version":{doc.get('conf_upload_version', 'N/A')},
                "last_changed_date":{doc.get('last_changed_date', 'N/A')},
                "secondary_product":{doc.get('secondary_product', 'N/A')},
                "extracted_source_ids":{doc.get('extracted_source_ids', 'N/A')}
            """.strip()
            # Split into chunks
            chunks = text_splitter.split_text(content)
            logger.info(f"Document {doc_idx + 1}: Split into {len(chunks)} chunks")
            
            # Create Document objects for each chunk
            for chunk in chunks:
                doc_id = str(uuid.uuid4())
                metadata = prepare_safe_metadata(doc, doc_id, formatted_conf_name)
                all_doc_objects.append(Document(page_content=chunk, metadata=metadata))
        
        # Create optimized batches
        optimized_batches = create_optimized_batches(all_doc_objects, max_batch_size_mb)
        logger.info(f"Created {len(optimized_batches)} optimized batches for processing")
        
        # Process each batch
        for batch_idx, batch_docs in enumerate(optimized_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(optimized_batches)} ({len(batch_docs)} chunks)")
            
            # Generate embeddings for this batch
            batch_texts = [doc.page_content for doc in batch_docs]
            logger.info(f"Generating embeddings for {len(batch_texts)} chunks")
            
            try:
                batch_embeddings = embedding_model.embed_documents(batch_texts)
                logger.info(f"Successfully generated {len(batch_embeddings)} embeddings")
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {batch_idx + 1}: {e}")
                continue
            
            # Prepare vectors for upsert
            vectors = []
            chunk_ids = []
            
            for j, (doc, embedding_values) in enumerate(zip(batch_docs, batch_embeddings)):
                vector_id = str(uuid.uuid4())
                chunk_ids.append(vector_id)
                
                # Include the page_content in the metadata (truncated)
                metadata = doc.metadata.copy()
                # Truncate page_content to avoid size issues
                metadata["page_content"] = doc.page_content[:400] if len(doc.page_content) > 400 else doc.page_content
                
                vectors.append((vector_id, embedding_values, metadata))
            
            # Estimate batch size before upsert
            estimated_size = estimate_batch_size_mb(vectors)
            logger.info(f"Batch {batch_idx + 1} estimated size: {estimated_size:.2f}MB")
            
            # If batch is still too large, split it further
            if estimated_size > max_batch_size_mb:
                logger.warning(f"Batch {batch_idx + 1} too large ({estimated_size:.2f}MB), splitting further")
                
                # Split the batch in half and process separately
                mid_point = len(vectors) // 2
                sub_batches = [vectors[:mid_point], vectors[mid_point:]]
                
                for sub_batch_idx, sub_vectors in enumerate(sub_batches):
                    if sub_vectors:  # Only process if not empty
                        try:
                            logger.info(f"Upserting sub-batch {sub_batch_idx + 1}/2 with {len(sub_vectors)} vectors")
                            index.upsert(vectors=sub_vectors, namespace=namespace)
                            logger.info(f"Successfully upserted sub-batch {sub_batch_idx + 1}/2")
                        except Exception as e:
                            logger.error(f"Error upserting sub-batch {sub_batch_idx + 1}/2: {e}")
            else:
                # Upsert the batch
                try:
                    logger.info(f"Upserting batch {batch_idx + 1} with {len(vectors)} vectors to namespace '{namespace}'")
                    index.upsert(vectors=vectors, namespace=namespace)
                    logger.info(f"Successfully upserted batch {batch_idx + 1}")
                except Exception as e:
                    logger.error(f"Error upserting batch {batch_idx + 1}: {e}")
                    continue
            
            # Add IDs to the global list
            document_ids.extend(chunk_ids)
            processed_count += len(batch_docs)
            
            # Update job status
            stats.update_processed(len(batch_docs))
            current_stats = stats.get_stats()
            background_jobs[job_id].update(current_stats)
            background_jobs[job_id]["status"] = f"Processed {processed_count} chunks in {batch_idx + 1}/{len(optimized_batches)} batches"
            
            logger.info(f"Progress: {processed_count}/{len(all_doc_objects)} chunks processed")
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)
        
        # Job completed successfully
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["document_ids"] = document_ids
        background_jobs[job_id]["count"] = len(document_ids)
        
        completion_message = f"Successfully processed {len(document_ids)} chunks from {len(all_documents)} documents"
        if failed_pages:
            completion_message += f" (failed to fetch {len(failed_pages)} pages)"
        
        background_jobs[job_id]["message"] = completion_message
        logger.info(f"Job {job_id} completed: {completion_message}")
        
    except Exception as e:
        error_message = f"Job {job_id} failed: {str(e)}"
        logger.error(error_message, exc_info=True)
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["message"] = str(e)

@app.post("/api/ingest_conference_data", response_model=EmbeddingResponse, dependencies=[Depends(verify_api_key)])
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
    try:
        request.conferenceName, request.conferenceIteration = extract_acronym_data(request.conferenceAcronym)
        
        logger.info(f"Received request to ingest conference data: {request.conferenceName} {request.conferenceIteration}")
        
        # Generate a job ID
        job_id = str(uuid.uuid4())
        logger.info(f"Generated job ID: {job_id}")
        
        namespace = f"{request.conferenceName}_{request.conferenceIteration}".lower().replace(" ", "_")
        logger.info(f"Using namespace: {namespace}")
        
        # Initialize job status
        background_jobs[job_id] = {
            "status": "initializing",
            "processed_items": 0,
            "total_items": 0,
            "current_page": 0,
            "total_pages": 0,
            "failed_pages": 0,
            "namespace": namespace
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
            request.max_parallel_requests,
            request.batch_size,
            request.max_batch_size_mb,
            job_id,
            pc,
            config,
            request.timeStamp
        )
        
        success_message = f"Started ingestion of conference data for {request.conferenceName} {request.conferenceIteration} with parallel processing"
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
    conferenceAcronym: str
    conferenceName: str = ""
    conferenceIteration: str = ""
    query: str
    top_k: int = 10
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
Try to go deep into the abstracts and identify all necessary information for the given query and summarize well, please try to more analytical and to the point as you are givinig information to medical people who will be in hurry and required to get information in a concise and efficient manner, and highlight key aspect of the answer by **bolding** it and follow the markdown format for efficient formatting, and stick to given data only, do not add details from other sources.
And Try to sound like a helper which has good technical know how but also great at communicating complex stuff in easier way for people.
Always give answer with trial-ids of the abstracts you are referring to, so that user can refer to the abstracts for more details if required.
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

# Constants for hybrid RAG
METADATA_FIELDS = [
    "Title", "conferenceDay", "conferenceDate", "phase", "score", "source", "status",
    "authors", "discussedDisease", "is_paid", "sponser", "category", "end_time", 
    "start_time", "location", "biomarker", "duplicate", "news_type", "paperpresenter",
    "redtag_id", "source_id", "trail_ids", "session_id", "institution", "source_type",
    "investigator", "journal_name", "session_type", "sub_category", "session_title",
    "abstract_number", "generation_date", "conference_source", "conf_upload_version",
    "last_changed_date", "is_product_owner_sponsered_active", "news_item_id"
]

QUANTITATIVE_PATTERNS = [
    r'^(which|what)\s+.*\b(sessions?|presentations?|talks?|speakers?|topics?|categories?)\b',
    r'^(how\s+many|how\s+much)\b',
    r'\b(count|number\s+of|total|all\s+the|list\s+all)\b',
    r'\b(sessions?\s+on|presentations?\s+about|talks?\s+related\s+to)\b'
]

# Default RAG prompt
prompt_template = """You are an expert medical orator who analyzes abstracts from medical conferences to answer user questions.
If the abstracts contain ANY information related to the question, even if incomplete, summarize what is available, and always stick to the given abstracts. Do not provide any information beyond what's in the abstracts.
Only say you cannot answer if there is absolutely no relevant information. Present your answer in bullet points for easier understanding.
Try to go deep into the abstracts and identify all necessary information for the given query and summarize well, please try to more analytical and to the point as you are givinig information to medical people who will be in hurry and required to get information in a concise and efficient manner, and highlight key aspect of the answer by **bolding** it and follow the markdown format for efficient formatting, and stick to given data only, do not add details from other sources.
And Try to sound like a helper which has good technical know how but also great at communicating complex stuff in easier way for people.
Always give answer with abstract-number of the abstracts you are referring to, so that user can refer to the abstracts for more details if required.
QUESTION: {question}

RETRIEVED ABSTRACTS:
{context}

ANSWER:
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# Enhanced models for hybrid RAG
class ConferenceQueryRequest(BaseModel):
    conferenceAcronym: str = Field(..., description="Conference acronym (e.g., 'asco-gu-2025')")
    query: str = Field(..., min_length=1, max_length=1000, description="The search query")
    conferenceName: str = ""
    conferenceIteration: str = ""
    use_rag: bool = Field(default=True, description="Whether to use RAG for response generation")
    model_name: str = Field(default="gpt-4o-mini", description="LLM model to use for generation")
    temperature: float = Field(default=0, ge=0.0, le=1.0, description="Temperature for LLM")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of documents to retrieve")
    save_result: bool = Field(default=False, description="Whether to save the result")
    use_hybrid: bool = Field(default=True, description="Whether to use hybrid RAG approach")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="The search query")
    save_result: bool = Field(default=False, description="Whether to save the result to storage")
    top_k: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return")

class RagResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]] = []
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float

class QueryResponseFilter(BaseModel):
    query_type: str
    explanation: str
    response: str
    original_query: str
    filters_applied: Optional[Dict[str, Any]] = None
    search_terms: Optional[List[str]] = None
    is_counting_query: Optional[bool] = None
    expected_fields: Optional[List[str]] = None
    total_results_found: Optional[int] = None
    processing_time_ms: Optional[float] = None
    timestamp: str

class StreamingCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tokens = []
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)
        self.text += token

# Service classes
class ResultStorage:
    """Thread-safe result storage with enhanced error handling."""
    
    def __init__(self, storage_file: str = "query_results.json"):
        self.storage_file = storage_file
        self.results = self._load_results()
        self._lock = threading.Lock()
    
    def _load_results(self) -> List[Dict]:
        """Load existing results from JSON file with error handling."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Error loading results: {e}")
        return []
    
    def _save_results(self):
        """Save results to JSON file with error handling."""
        try:
            # Create backup before saving
            if os.path.exists(self.storage_file):
                backup_file = f"{self.storage_file}.backup"
                with open(self.storage_file, 'r') as original:
                    with open(backup_file, 'w') as backup:
                        backup.write(original.read())
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def add_result(self, query_result: Dict[str, Any]):
        """Add a new query result to storage with thread safety."""
        try:
            with self._lock:
                stored_result = {
                    "timestamp": datetime.now().isoformat(),
                    "query": query_result.get("original_query", ""),
                    "query_type": query_result.get("query_type", ""),
                    "explanation": query_result.get("explanation", ""),
                    "response": query_result.get("response", ""),
                    "filter_type": self._determine_filter_type(query_result),
                    "filters_applied": query_result.get("filters_applied", {}),
                    "search_terms": query_result.get("search_terms", []),
                    "is_counting_query": query_result.get("is_counting_query", False),
                    "total_results_found": self._get_total_results(query_result),
                    "fallback_used": query_result.get("fallback", False),
                    "processing_time_ms": query_result.get("processing_time_ms", 0),
                }
                
                self.results.append(stored_result)
                self._save_results()
                logger.info(f"Result saved to {self.storage_file}")
            
        except Exception as e:
            logger.error(f"Error adding result: {e}")
            raise
    
    def _determine_filter_type(self, query_result: Dict[str, Any]) -> str:
        """Determine what type of filtering was used."""
        if query_result.get("query_type") == "FILTER_SEARCH":
            filters = query_result.get("filters_applied", {})
            search_terms = query_result.get("search_terms", [])
            
            if filters and search_terms:
                return "HYBRID_FILTER_VECTOR"
            elif filters:
                return "METADATA_FILTER"
            elif search_terms:
                return "VECTOR_WITH_TERMS"
            else:
                return "FILTER_SEARCH_NO_CONSTRAINTS"
        elif query_result.get("query_type") == "VECTOR_SEARCH":
            return "PURE_VECTOR_SEARCH"
        else:
            return "GENERAL_LLM"
    
    def _get_total_results(self, query_result: Dict[str, Any]) -> int:
        """Extract total number of results found."""
        if "analysis" in query_result:
            return query_result["analysis"].get("count", 0)
        elif "retrieved_docs" in query_result:
            return len(query_result["retrieved_docs"])
        else:
            return 0

class QueryClassifier:
    """Enhanced query classifier with better filter detection."""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.quantitative_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in QUANTITATIVE_PATTERNS]
    
    def is_quantitative_query(self, query: str) -> bool:
        """Check if query is quantitative/filtering type."""
        return any(pattern.search(query) for pattern in self.quantitative_patterns)
    
    async def classify_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query classification with better filter detection."""
        is_quantitative = self.is_quantitative_query(query)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"""
                    You are a query classifier for a medical conference information system.
                    
                    Available metadata fields: {', '.join(METADATA_FIELDS)}
                    
                    Classification Rules:
                    1. FILTER_SEARCH: Use when query:
                       - Starts with "which", "how many", "how much"
                       - Contains counting/listing words: "count", "number of", "total", "all the", "list all"
                       - Asks for specific filtering by metadata fields (date, category, disease, etc.)
                       - Requires structured data retrieval or quantitative analysis
                    
                    2. VECTOR_SEARCH: Use when query:
                       - Asks about content/meaning without specific constraints
                       - Requires semantic understanding of text content
                       - Asks "what is", "explain", "describe" about concepts
                                        
                    For FILTER_SEARCH queries:
                    - Extract specific metadata constraints as exact values or lists of possible values
                    - For partial matching (like "genetics"), provide a list of related terms
                    - Handle date ranges, categories, disease names, etc.
                    - Create a simplified query for semantic search after filtering
                    - Identify if it's a counting/listing query vs content query
                    
                    Important: For field values that need partial matching, provide multiple possible exact matches instead of using regex patterns.
                    
                    Query to classify: "{query}"
                    Detected as quantitative: {is_quantitative}
                    
                    Return JSON with:
                    - "query_type": "VECTOR_SEARCH", "FILTER_SEARCH"
                    - "explanation": Why this classification was chosen
                    - "filters": Dictionary of metadata filters with exact values or $in arrays
                    - "rewritten_query": Simplified query after removing constraints
                    - "is_counting_query": Boolean indicating if it's asking for counts/lists
                    - "expected_fields": List of fields to focus on in response
                    - "search_terms": List of terms to search for in vector space if filtering fails
                    """},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            # Fallback classification
            return {
                "query_type": "VECTOR_SEARCH",
                "explanation": "Fallback classification due to API error",
                "filters": {},
                "rewritten_query": query,
                "is_counting_query": is_quantitative,
                "expected_fields": [],
                "search_terms": query.split()
            }

class HybridRAG:
    """Enhanced Hybrid RAG system with improved filter search and result storage."""
    
    def __init__(self, openai_client: OpenAI, pinecone_index, embed_model: str = "text-embedding-3-large", storage_file: str = "query_results.json"):
        self.client = openai_client
        self.index = pinecone_index
        self.embed_model = embed_model
        self.query_classifier = QueryClassifier(openai_client)
        self.storage = ResultStorage(storage_file)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=self.embed_model,
                input=text,
                dimensions=3072
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")
    
    def vector_search(self, query: str, namespace: str, top_k: int = 5) -> List[Dict]:
        """Perform vector-based semantic search."""
        try:
            query_embedding = self.embed_text(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            return results.matches
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise HTTPException(status_code=500, detail="Vector search failed")
    
    def build_pinecone_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert filters to Pinecone filter format using only supported operators."""
        filter_dict = {}
        logger.info(f"Building Pinecone filter from: {filters}")
        for key, value in filters.items():
            if key not in METADATA_FIELDS:
                continue
            logger.info(f"Processing filter for field: {key} with value: {value}")
            if isinstance(value, str):
                filter_dict[key] = {"$eq": value}
            elif isinstance(value, list):
                filter_dict[key] = {"$in": value}
            elif isinstance(value, dict):
                supported_ops = {"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"}
                filtered_value = {k: v for k, v in value.items() if k in supported_ops}
                if filtered_value:
                    filter_dict[key] = filtered_value
        
        return filter_dict
    
    def filter_search_with_fallback(self, query: str, namespace: str, filters: Dict[str, Any], search_terms: List[str] = None, top_k: int = 50) -> List[Dict]:
        """Enhanced filtered search with fallback to vector search."""
        results = []
        print("filters:", filters)
        print("search_terms:", search_terms)
        logger.info(f"Starting filtered search with query: '{query}', namespace: '{namespace}', filters: {filters}, search_terms: {search_terms}, top_k: {top_k}")
        # Try 1: Filtered search with exact matches
        if filters:
            try:
                pinecone_filter = self.build_pinecone_filter(filters)
                query_embedding = self.embed_text(query) if query.strip() else self.embed_text(" ".join(search_terms or []))
                logger.info(f"Generated embedding for query: {query_embedding[:10]}... (truncated)")
                logger.info(f"Trying filter search with filters: {pinecone_filter}")
                if pinecone_filter:
                    logger.info(f"Trying filter search with: {pinecone_filter}")
                    results = self.index.query(
                        vector=query_embedding,
                        filter=pinecone_filter,
                        top_k=top_k,
                        include_metadata=True,
                        namespace=namespace
                    ).matches
                    print("Filter search results:", results)
                    if results:
                        logger.info(f"Filter search returned {len(results)} results")
                        return results
                    
            except Exception as e:
                logger.error(f"Filter search failed: {e}")
        
        # Try 2: Vector search with search terms
        if search_terms:
            try:
                search_query = " ".join(search_terms)
                logger.info(f"Trying vector search with terms: {search_query}")
                results = self.vector_search(search_query, namespace, top_k=5000)
                
                if results:
                    logger.info(f"Vector search returned {len(results)} results")
                    filtered_results = []
                    for doc in results:
                        doc_text = str(doc.get('metadata', {})).lower()
                        if any(term.lower() in doc_text for term in search_terms):
                            filtered_results.append(doc)
                    
                    if filtered_results:
                        logger.info(f"Manual filtering returned {len(filtered_results)} results")
                        return filtered_results
                    else:
                        logger.info("Manual filtering returned no results, using all vector results")
                        return results
                        
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
        
        # Try 3: Simple vector search with original query
        try:
            logger.info("Trying simple vector search with original query")
            results = self.vector_search(query, namespace, top_k=top_k)
            logger.info(f"Simple vector search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Simple vector search failed: {e}")
            return []
    
    def analyze_filtered_results(self, results: List[Dict], classification: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze filtered results for counting and aggregation queries."""
        if not results:
            return {"count": 0, "items": [], "summary": {}}
        
        analysis = {"count": len(results), "items": []}
        expected_fields = classification.get("expected_fields", [])
        all_metadata = [doc['metadata'] for doc in results]
        
        # Create summary statistics
        summary = {}
        for field in METADATA_FIELDS:
            if field in expected_fields or not expected_fields:
                field_values = []
                for meta in all_metadata:
                    value = meta.get(field, '')
                    if value:
                        if isinstance(value, list):
                            field_values.extend(value)
                        else:
                            field_values.append(str(value))
                
                if field_values:
                    if field in ["date", "start_time", "end_time"]:
                        summary[field] = {"unique_values": list(set(field_values))}
                    else:
                        counter = Counter(field_values)
                        summary[field] = {
                            "unique_count": len(counter),
                            "top_values": counter.most_common(5)
                        }
        
        # Create structured items list
        for doc in results:
            item = {
                "id": doc.get('id', ''),
                "score": doc.get('score', 0),
                "metadata": doc.get('metadata', {})
            }
            analysis["items"].append(item)
        
        analysis["summary"] = summary
        return analysis
    
    def generate_filter_response(self, analysis: Dict[str, Any], query: str, classification: Dict[str, Any]) -> str:
        """Generate response for filter-based queries with counting/listing focus."""
        count = analysis["count"]
        
        if count == 0:
            return "I couldn't find any items matching your criteria in the conference data."
        
        # Build context for LLM
        context_parts = [f"Total results found: {count}"]
        
        # Add summary statistics
        if analysis["summary"]:
            context_parts.append("Summary of results:")
            for field, stats in analysis["summary"].items():
                if "top_values" in stats and stats["top_values"]:
                    top_items = [f"{item[0]}: {item[1]}" for item in stats["top_values"][:3]]
                    context_parts.append(f"- {field}: {', '.join(top_items)}")
                elif "unique_values" in stats and stats["unique_values"]:
                    unique_vals = stats["unique_values"][:3]
                    context_parts.append(f"- {field}: {', '.join(unique_vals)}")
        
        # Add sample items
        sample_items = analysis["items"][:5]
        if sample_items:
            context_parts.append("Sample results:")
            for i, item in enumerate(sample_items, 1):
                meta = item["metadata"]
                title = meta.get("title", "Untitled")
                category = meta.get("category", "Unknown category")
                sub_category = meta.get("sub_category", "")
                display_title = f"{title} ({category}" + (f" - {sub_category}" if sub_category else "") + ")"
                context_parts.append(f"{i}. {display_title}")
        
        context = "\n".join(context_parts)
        
        # Generate response using LLM
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """
                    You are analyzing conference data results. Provide a clear, informative response based on the analysis provided. 
                    
                    For counting queries ("how many", "how much"), focus on:
                    - The total count prominently
                    - Key breakdowns and categories
                    - Notable patterns or concentrations
                    
                    For "which" queries, focus on:
                    - Listing the relevant items clearly
                    - Categorizing or grouping them logically
                    - Highlighting key information
                    
                    Be specific and use the data provided. Format your response clearly with bullet points or numbers when appropriate.
                    Always start with the direct answer to the question.
                    """},
                    {"role": "user", "content": f"Original query: {query}\n\nAnalysis results:\n{context}\n\nProvide a comprehensive answer:"}
                ]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating filter response: {e}")
            return f"Found {count} results matching your criteria, but couldn't generate detailed analysis."
    
    def general_llm_response(self, query: str) -> str:
        """Get a response from the LLM for general knowledge questions."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant knowledgeable about oncology and medical conferences."},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            return "I apologize, but I'm unable to provide a response at this time due to technical difficulties."

    def format_abstracts_for_medical_prompt(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents specifically for the medical expert prompt."""
        if not retrieved_docs:
            return "No relevant abstracts found."
        
        formatted_abstracts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get('metadata', {})
            content = metadata.get('page_content', 'Content not available')
            
            # Extract key metadata for medical context
            trial_id = metadata.get('trial_ids', 'N/A')
            title = metadata.get('title', 'Title not available')
            authors = metadata.get('authors', 'Authors not available')
            category = metadata.get('category', 'N/A')
            disease_names = metadata.get('disease_names', 'N/A')
            sponsor = metadata.get('sponsor', 'N/A')
            
            abstract_entry = f"""
                **ABSTRACT {i}:**
                - **Trial ID:** {trial_id}
                - **Title:** {title}
                - **Authors:** {authors}
                - **Category:** {category}
                - **Disease/Indication:** {disease_names}
                - **Sponsor:** {sponsor}
                - **Content:** {content}
                            """.strip()
            
            formatted_abstracts.append(abstract_entry)
        
        return "\n\n".join(formatted_abstracts)
    def generate_rag_response(self, retrieved_docs: List[Dict], query: str) -> str:
        """Generate a response using the medical expert prompt template."""
        if not retrieved_docs:
            return "I couldn't find relevant abstracts in the conference data to answer your question."
        
        # Format abstracts for the medical expert prompt
        # formatted_context = self.format_abstracts_for_medical_prompt(retrieved_docs)
        logger.info(f"Retrieved {(retrieved_docs)} documents for RAG response")
        formatted_context = retrieved_docs
        
        try:
            # Use the custom medical expert prompt
            formatted_prompt = PROMPT.format(
                context=formatted_context,
                question=query
            )
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert medical orator analyzing conference abstracts. Follow the given prompt template exactly."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0 # Lower temperature for more consistent medical responses
            )
            logger.info(f"Generated RAG response: {response.choices[0].message.content[:100]}... (truncated)")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return "I found relevant abstracts but couldn't generate a proper analysis due to technical difficulties. Please try again."
 
    async def process_query(self, query: str, namespace: str, save_result: bool = True, top_k: int = 10) -> Dict[str, Any]:
        """Enhanced query processing with result storage."""
        start_time = datetime.now()
        logger.info(f"Processing query: {query}")
        
        try:
            # Classify the query
            classification = await self.query_classifier.classify_query(query)
            query_type = classification["query_type"]
            
            logger.info(f"Query classified as: {query_type}")
            
            result = {
                "query_type": query_type,
                "explanation": classification["explanation"],
                "original_query": query
            }
            
            # Route the query based on classification
            if query_type == "FILTER_SEARCH":
                rewritten_query = classification.get("rewritten_query", "")
                filters = classification.get("filters", {})
                search_terms = classification.get("search_terms", [])
                is_counting = classification.get("is_counting_query", False)
                
                result.update({
                    "filters_applied": filters,
                    "search_terms": search_terms,
                    "is_counting_query": is_counting,
                    "expected_fields": classification.get("expected_fields", [])
                })
                
                # Perform filtered search with fallback
                retrieved = self.filter_search_with_fallback(
                    rewritten_query, namespace, filters, search_terms, 
                    top_k=min(50, top_k * 5) if is_counting else top_k
                )
                
                if retrieved:
                    analysis = self.analyze_filtered_results(retrieved, classification)
                    result["analysis"] = analysis
                    
                    if is_counting or query.lower().startswith(('which', 'how many', 'how much')):
                        result["response"] = self.generate_filter_response(analysis, query, classification)
                    else:
                        result["response"] = self.generate_rag_response(retrieved, query)
                    
                    result["total_results_found"] = len(retrieved)
                else:
                    result["response"] = "I couldn't find any information matching your specific criteria in the conference data."
                    result["fallback"] = True
                    result["total_results_found"] = 0
                    
            elif query_type == "VECTOR_SEARCH":
                retrieved = self.vector_search(query, namespace, top_k)
                if retrieved:
                    result["response"] = self.generate_rag_response(retrieved, query)
                    result["total_results_found"] = len(retrieved)
                else:
                    result["response"] = self.general_llm_response(query)
                    result["fallback"] = True
                    result["total_results_found"] = 0
                    
            else:  # GENERAL
                result["response"] = self.general_llm_response(query)
                result["total_results_found"] = 0
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result["processing_time_ms"] = processing_time
            
            # Save result to JSON storage
            if save_result:
                self.storage.add_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            error_result = {
                "query_type": "ERROR",
                "explanation": f"Error occurred during processing: {str(e)}",
                "original_query": query,
                "response": "I apologize, but I encountered an error while processing your query. Please try again.",
                "processing_time_ms": processing_time,
                "total_results_found": 0
            }
            
            if save_result:
                try:
                    self.storage.add_result(error_result)
                except:
                    pass  # Don't fail completely if storage fails
            
            return error_result

# Global variables for services
openai_client = None
pinecone_index = None
hybrid_rag = None

async def initialize_services():
    """Initialize OpenAI and Pinecone clients."""
    global openai_client, pinecone_index, hybrid_rag
    
    try:
        # Validate environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "conference-data-production"
        pinecone_index = pc.Index(index_name)
        
        # Initialize HybridRAG
        hybrid_rag = HybridRAG(
            openai_client=openai_client,
            pinecone_index=pinecone_index,
            storage_file="conference_query_results.json"
        )
        
        logger.info("Services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

async def cleanup_services():
    """Cleanup services on shutdown."""
    logger.info("Cleaning up services...")

async def get_hybrid_rag():
    if hybrid_rag is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return hybrid_rag

# Background jobs tracking
background_jobs = {}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    await initialize_services()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await cleanup_services()

# Enhanced query endpoint with streaming support
@app.post("/api/conference/query_streaming", dependencies=[Depends(verify_api_key)])
async def query_conference_data_streaming(
    request: ConferenceQueryRequest,
    response: fastapi.Response,
    background_tasks: BackgroundTasks,
    pc: Pinecone = Depends(get_pinecone_client),
    config: Config = Depends(get_config)
):
    """
    Query conference data with enhanced streaming and hybrid RAG capability.
    """
    try:    
        # Extract conference information
        conferenceAcronym = request.conferenceAcronym
        request.conferenceName, request.conferenceIteration = extract_acronym_data(conferenceAcronym)
        logger.info(f"Processing conference query: '{request.query}' for {request.conferenceName} {request.conferenceIteration}")
        
        # Create namespace from conference name and iteration
        namespace = f"{request.conferenceName.lower()}_{request.conferenceIteration}"
        index_name = "conference-data-production"
        
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
        
        # Use hybrid RAG if requested
        if request.use_hybrid and hybrid_rag:
            logger.info("Using Hybrid RAG approach")
            
            # Process query using hybrid RAG
            result = await hybrid_rag.process_query(
                query=request.query,
                namespace=namespace,
                save_result=request.save_result,
                top_k=request.top_k
            )
            
            # If using RAG, stream the response
            if request.use_rag:
                response.headers["Content-Type"] = "text/plain"
                response.headers["Cache-Control"] = "no-cache"
                response.headers["Connection"] = "keep-alive"
                
                async def generate_stream():
                    # Stream the hybrid RAG response
                    response_text = result.get("response", "No response available")
                    
                    # Stream word by word for better UX
                    words = response_text.split()
                    for i, word in enumerate(words):
                        yield word + (" " if i < len(words) - 1 else "")
                        # await asyncio.sleep(0.01)  # Small delay for streaming effect
                
                return fastapi.responses.StreamingResponse(
                    generate_stream(),
                    media_type="text/plain"
                )
            else:
                # Return JSON response for non-streaming hybrid RAG
                return QueryResponseFilter(
                    query_type=result.get("query_type", "HYBRID"),
                    explanation=result.get("explanation", "Hybrid RAG processing"),
                    response=result.get("response", ""),
                    original_query=request.query,
                    filters_applied=result.get("filters_applied"),
                    search_terms=result.get("search_terms"),
                    is_counting_query=result.get("is_counting_query"),
                    expected_fields=result.get("expected_fields"),
                    total_results_found=result.get("total_results_found"),
                    processing_time_ms=result.get("processing_time_ms"),
                    timestamp=datetime.now().isoformat()
                )
        
        # Original RAG implementation if hybrid is not used
        if request.use_rag:
            logger.info("Setting up traditional RAG system with LangChain")
            
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
                
                # Store in result storage if requested
                if request.save_result and hybrid_rag:
                    try:
                        hybrid_rag.storage.add_result({
                            "query_type": "TRADITIONAL_RAG",
                            "explanation": "Traditional RAG with streaming",
                            "original_query": request.query,
                            "response": accumulated_answer,
                            "processing_time_ms": total_time * 1000,
                            "total_results_found": len(relevant_docs)
                        })
                    except Exception as e:
                        logger.error(f"Error storing result: {e}")
            
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
        logger.error(f"Error in query_conference_data_streaming: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

######## update functionality

# Enhanced update functionality with parallel processing and optimized batching
async def update_conference_embeddings(
    conference_name: str,
    conference_iteration: str,
    timeStamp: datetime,
    index_name: str,
    namespace: str,
    chunk_size: int,
    chunk_overlap: int,
    max_pages: int,
    max_parallel_requests: int,
    max_batch_size_mb: float,
    job_id: str,
    pc: Pinecone,
    config: Config
):
    """Background task to update conference data embeddings only if version has changed - with parallel processing"""
    stats = ProcessingStats()
    
    try:
        logger.info(f"Starting update job {job_id} for conference {conference_name} {conference_iteration} at timestamp {timeStamp}")
        
        background_jobs[job_id] = {
            "status": "running", 
            "processed_items": 0, 
            "total_items": 0,
            "updated_items": 0,
            "skipped_items": 0,
            "current_page": 0,
            "total_pages": 0,
            "failed_pages": 0
        }
        
        # Format the conference name for the API
        formatted_conf_name = f"{conference_name} {conference_iteration}"
        logger.info(f"Formatted conference name: {formatted_conf_name}")
        
        # Fetch first page to get total pages
        logger.info(f"Fetching first page to determine total pages")
        _, total_pages = fetch_conference_data_page(formatted_conf_name, 1, timeStamp)
        
        # Limit total pages to max_pages
        total_pages = min(total_pages, max_pages)
        logger.info(f"Will process {total_pages} pages (limited by max_pages={max_pages})")
        
        stats.set_totals(0, total_pages)
        background_jobs[job_id]["total_pages"] = total_pages
        background_jobs[job_id]["status"] = f"Fetching data from {total_pages} pages in parallel"
        
        # Fetch all pages in parallel
        logger.info(f"Starting parallel fetch of {total_pages} pages with {max_parallel_requests} concurrent requests")
        all_documents, failed_pages = fetch_conference_data_parallel(
            formatted_conf_name, 
            total_pages, 
            timeStamp, 
            max_parallel_requests,
            stats
        )
        
        # Update job status with fetch results
        total_document_count = len(all_documents)
        stats.set_totals(total_document_count, total_pages)
        
        background_jobs[job_id].update({
            "total_items": total_document_count,
            "failed_pages": len(failed_pages),
            "status": f"Processing {total_document_count} documents for updates"
        })
        
        logger.info(f"Total documents collected: {total_document_count}")
        
        if failed_pages:
            logger.warning(f"Failed to fetch {len(failed_pages)} pages: {failed_pages}")
        
        # Create index if it doesn't exist
        logger.info(f"Checking if index {index_name} exists")
        if not pc.has_index(index_name):
            logger.info(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(10)  # Wait for index to be ready
        else:
            logger.info(f"Index {index_name} already exists")
        
        # Get index
        index = pc.Index(index_name)
        
        # Initialize text splitter and embedding model
        logger.info(f"Initializing text splitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        logger.info("Initializing OpenAI embedding model (text-embedding-3-large)")
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large", 
            openai_api_key=config.openai_api_key
        )
        
        document_ids = []
        processed_count = 0
        updated_count = 0
        skipped_count = 0
        
        # Process documents in optimized batches
        logger.info(f"Processing documents in optimized batches")
        
        # Create a list to track documents that need updating
        documents_to_update = []
        
        # First pass: Check which documents need updates
        for doc_idx, doc in enumerate(all_documents):
            source_id = str(doc.get("source_id", ""))
            if not source_id:
                logger.warning(f"Document at index {doc_idx} has no source_id, skipping")
                continue
            
            # Get the new Conf_Upload_Version
            new_version = str(doc.get("Conf_Upload_Version", ""))
            
            try:
                # Check if document already exists in the index
                existing_docs = index.query(
                    vector=[0] * 3072,  # Dummy vector for metadata-only query
                    namespace=namespace,
                    filter={"source_id": source_id},
                    include_metadata=True,
                    top_k=1
                )
                
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
                
                # Add to update list
                documents_to_update.append((doc, source_id, new_version, existing_docs))
                
            except Exception as e:
                logger.error(f"Error checking document {source_id}: {e}")
                # Add to update list anyway to be safe
                documents_to_update.append((doc, source_id, new_version, None))
            
            processed_count += 1
            
            # Update job status periodically
            if processed_count % 10 == 0:
                background_jobs[job_id].update({
                    "processed_items": processed_count,
                    "updated_items": updated_count,
                    "skipped_items": skipped_count,
                    "status": f"Checking {processed_count}/{len(all_documents)} documents for updates"
                })
        
        logger.info(f"Found {len(documents_to_update)} documents that need updating")
        
        # Second pass: Process documents that need updates
        if documents_to_update:
            # Convert documents to Document objects for batch processing
            all_doc_objects = []
            source_id_mapping = {}  # Map document objects to source_ids for cleanup
            
            for doc, source_id, new_version, existing_docs in documents_to_update:
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
                    Start Time: {doc.get('start_time', 'N/A')} 
                    End Time: {doc.get('end_time', 'N/A')}
                    Location: {doc.get('location', 'N/A')}
                    Category: {doc.get('category', 'N/A')}
                    Sub Category: {doc.get('sub_category', 'N/A')}
                    Diseases: {disease_info}
                    Sponsor: {doc.get('sponsor', 'N/A')}
                    Session Text: {doc.get('session_text', 'N/A')}
                    Disclosure: {doc.get('disclosures', 'N/A')}
                    News Type: {doc.get('news_type', 'N/A')}
                    Affiliation: {doc.get('affiliations', 'N/A')}
                    Details: {doc.get('details', 'N/A')}
                    Summary: {doc.get('summary', 'N/A')}
                    Authors: {doc.get('authors', 'N/A')}
                    Abstract Number: {doc.get('abstract_number', 'N/A')}
                    Source Type: {doc.get('source_type', 'N/A')}
                    Session Type: {doc.get('session_type', 'N/A')}
                    Institution: {doc.get('institution', 'N/A')}
                    Brief Title: {doc.get('brief_title', 'N/A')}
                    Journal Name: {doc.get('journal_name', 'N/A')}
                    Investigator: {doc.get('investigator', 'N/A')}
                    KOL: {doc.get('kol', 'N/A')}
                """.strip()
                
                # Split into chunks
                chunks = text_splitter.split_text(content)
                logger.info(f"Document with source_id {source_id}: Split into {len(chunks)} chunks")
                
                # Create Document objects for each chunk
                for chunk in chunks:
                    doc_id = str(uuid.uuid4())
                    metadata = prepare_safe_metadata(doc, doc_id, formatted_conf_name)
                    
                    # Ensure Conf_Upload_Version is included in metadata
                    if new_version:
                        metadata["Conf_Upload_Version"] = new_version
                    
                    doc_obj = Document(page_content=chunk, metadata=metadata)
                    all_doc_objects.append(doc_obj)
                    source_id_mapping[id(doc_obj)] = (source_id, existing_docs)
            
            # Delete existing documents for sources being updated
            logger.info("Cleaning up existing documents for sources being updated")
            deletion_tasks = []
            
            def delete_existing_docs(source_id, existing_docs):
                try:
                    if existing_docs and existing_docs.matches:
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
                            return len(ids_to_delete)
                except Exception as e:
                    logger.error(f"Error deleting existing documents for source_id {source_id}: {str(e)}")
                return 0
            
            # Delete existing documents in parallel
            unique_sources = {}
            for doc, source_id, new_version, existing_docs in documents_to_update:
                if source_id not in unique_sources:
                    unique_sources[source_id] = existing_docs
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                deletion_futures = {
                    executor.submit(delete_existing_docs, source_id, existing_docs): source_id 
                    for source_id, existing_docs in unique_sources.items()
                }
                
                for future in as_completed(deletion_futures):
                    source_id = deletion_futures[future]
                    try:
                        deleted_count = future.result()
                        logger.info(f"Deleted {deleted_count} documents for source_id {source_id}")
                    except Exception as e:
                        logger.error(f"Error in deletion task for {source_id}: {e}")
            
            # Create optimized batches for new documents
            optimized_batches = create_optimized_batches(all_doc_objects, max_batch_size_mb)
            logger.info(f"Created {len(optimized_batches)} optimized batches for updating")
            
            # Process each batch
            for batch_idx, batch_docs in enumerate(optimized_batches):
                logger.info(f"Processing update batch {batch_idx + 1}/{len(optimized_batches)} ({len(batch_docs)} chunks)")
                
                # Generate embeddings for this batch
                batch_texts = [doc.page_content for doc in batch_docs]
                
                try:
                    batch_embeddings = embedding_model.embed_documents(batch_texts)
                    logger.info(f"Successfully generated {len(batch_embeddings)} embeddings for update")
                except Exception as e:
                    logger.error(f"Error generating embeddings for update batch {batch_idx + 1}: {e}")
                    continue
                
                # Prepare vectors for upsert
                vectors = []
                chunk_ids = []
                
                for j, (doc, embedding_values) in enumerate(zip(batch_docs, batch_embeddings)):
                    vector_id = str(uuid.uuid4())
                    chunk_ids.append(vector_id)
                    
                    # Include the page_content in the metadata (truncated)
                    metadata = doc.metadata.copy()
                    metadata["page_content"] = doc.page_content[:400] if len(doc.page_content) > 400 else doc.page_content

                    vectors.append((vector_id, embedding_values, metadata))
                
                # Estimate batch size before upsert
                estimated_size = estimate_batch_size_mb(vectors)
                logger.info(f"Update batch {batch_idx + 1} estimated size: {estimated_size:.2f}MB")
                
                # Handle oversized batches
                if estimated_size > max_batch_size_mb:
                    logger.warning(f"Update batch {batch_idx + 1} too large ({estimated_size:.2f}MB), splitting further")
                    
                    mid_point = len(vectors) // 2
                    sub_batches = [vectors[:mid_point], vectors[mid_point:]]
                    
                    for sub_batch_idx, sub_vectors in enumerate(sub_batches):
                        if sub_vectors:
                            try:
                                logger.info(f"Upserting update sub-batch {sub_batch_idx + 1}/2 with {len(sub_vectors)} vectors")
                                index.upsert(vectors=sub_vectors, namespace=namespace)
                                logger.info(f"Successfully upserted update sub-batch {sub_batch_idx + 1}/2")
                            except Exception as e:
                                logger.error(f"Error upserting update sub-batch {sub_batch_idx + 1}/2: {e}")
                else:
                    # Upsert the batch
                    try:
                        logger.info(f"Upserting update batch {batch_idx + 1} with {len(vectors)} vectors to namespace '{namespace}'")
                        index.upsert(vectors=vectors, namespace=namespace)
                        logger.info(f"Successfully upserted update batch {batch_idx + 1}")
                    except Exception as e:
                        logger.error(f"Error upserting update batch {batch_idx + 1}: {e}")
                        continue
                
                # Add IDs to the global list
                document_ids.extend(chunk_ids)
                
                # Update counters
                updated_count = len(documents_to_update)
                
                # Update job status
                background_jobs[job_id].update({
                    "processed_items": processed_count,
                    "updated_items": updated_count,
                    "skipped_items": skipped_count,
                    "status": f"Updated batch {batch_idx + 1}/{len(optimized_batches)} - Total: {updated_count} updated, {skipped_count} skipped"
                })
                
                logger.info(f"Update progress: batch {batch_idx + 1}/{len(optimized_batches)} completed")
                
                # Small delay to avoid overwhelming the system
                time.sleep(0.1)
        
        # Job completed successfully
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["document_ids"] = document_ids
        background_jobs[job_id]["count"] = len(document_ids)
        
        completion_message = f"Successfully processed {processed_count} documents. Updated {updated_count}, skipped {skipped_count}."
        if failed_pages:
            completion_message += f" (failed to fetch {len(failed_pages)} pages)"
        
        background_jobs[job_id]["message"] = completion_message
        logger.info(f"Update job {job_id} completed: {completion_message}")
        
    except Exception as e:
        error_message = f"Update job {job_id} failed: {str(e)}"
        logger.error(error_message, exc_info=True)
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["message"] = str(e)


@app.post("/api/update_conference_data", response_model=EmbeddingResponse, dependencies=[Depends(verify_api_key)])
async def update_conference_data(
    request: ConferenceDataRequest,
    background_tasks: BackgroundTasks,
    pc: Pinecone = Depends(get_pinecone_client),
    config: Config = Depends(get_config)
):
    """
    Update conference data embeddings, but only if the Conf_Upload_Version has changed.
    This will run as a background task and return a job ID for tracking progress.
    Enhanced with parallel processing and optimized batching.
    """
    try:
        conferenceAcronym = request.conferenceAcronym
        request.conferenceName, request.conferenceIteration = extract_acronym_data(conferenceAcronym)
        
        logger.info(f"Received request to update conference data: {request.conferenceName} {request.conferenceIteration}")
        
        # Generate a job ID
        job_id = str(uuid.uuid4())
        logger.info(f"Generated update job ID: {job_id}")
        
        namespace = f"{request.conferenceName}_{request.conferenceIteration}".lower().replace(" ", "_")
        logger.info(f"Using namespace: {namespace}")
        
        # Initialize job status
        background_jobs[job_id] = {
            "status": "initializing",
            "processed_items": 0,
            "total_items": 0,
            "updated_items": 0,
            "skipped_items": 0,
            "current_page": 0,
            "total_pages": 0,
            "failed_pages": 0,
            "namespace": namespace
        }
        
        # Start background task with enhanced parameters
        logger.info(f"Starting background update task for job {job_id}")
        background_tasks.add_task(
            update_conference_embeddings,
            request.conferenceName,
            request.conferenceIteration,
            request.timeStamp,
            request.index_name,
            namespace,
            request.chunk_size,
            request.chunk_overlap,
            request.max_pages,
            request.max_parallel_requests,
            request.max_batch_size_mb,
            job_id,
            pc,
            config
        )
        
        success_message = f"Started updating conference data for {request.conferenceName} {request.conferenceIteration} with parallel processing"
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
    
@app.post("/api/webhook/update_conference_data", response_model=EmbeddingResponse,dependencies=[Depends(verify_api_key)])
async def update_conference_data_webhook(
    request: ConferenceDataRequest,
    background_tasks: BackgroundTasks,
    pc: Pinecone = Depends(get_pinecone_client),
    config: Config = Depends(get_config)
):
    # acronym = request.conferenceAcronym.split("_")
    # print(acronym)
    # request.conferenceName = acronym[0] if len(acronym) > 0 else request.conferenceName
    # request.conferenceIteration = acronym[1] if len(acronym) > 1 else request.conferenceIteration
    conferenceAcronym = request.conferenceAcronym
    request.conferenceName,request.conferenceIteration = extract_acronym_data(conferenceAcronym)
    success_message = f"Started updating conference data for {request.conferenceName} {request.conferenceIteration}"
    logger.info(success_message)
    job_id = str(uuid.uuid4())
    logger.info(f"Generated update job ID: {job_id}")
        
    namespace = f"{request.conferenceName}_{request.conferenceIteration}".lower().replace(" ", "_")
    logger.info(f"Using namespace: {namespace}")
    background_tasks.add_task(
            update_conference_embeddings,
            request.conferenceName,
            request.conferenceIteration,
            request.timeStamp,
            request.index_name,
            namespace,
            request.chunk_size,
            request.chunk_overlap,
            request.max_pages,
            request.max_parallel_requests,
            request.max_batch_size_mb,
            job_id,
            pc,
            config
        )
    
    return EmbeddingResponse(
            success=True,
            message=success_message,
            job_id=str(uuid.uuid4()),  # Generate a unique job ID
            count=0
        )