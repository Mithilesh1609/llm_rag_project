from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
import os, re, json, time
import uuid
import requests
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from collections import Counter
import fastapi
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Header, status
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import threading
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

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
    """Extract acronym data from input string"""
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

app = FastAPI(title="Enhanced Vector Database API", 
              description="API for managing document embeddings and hybrid RAG queries")

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

# Enhanced models for hybrid RAG
class ConferenceQueryRequest(BaseModel):
    acronym: str = Field(..., description="Conference acronym (e.g., 'asco-gu-2025')")
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
        
        for key, value in filters.items():
            if key not in METADATA_FIELDS:
                continue
                
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
    
    def filter_search_with_fallback(self, query: str, namespace: str, filters: Dict[str, Any], search_terms: List[str] = None, top_k: int = 20) -> List[Dict]:
        """Enhanced filtered search with fallback to vector search."""
        results = []
        
        # Try 1: Filtered search with exact matches
        if filters:
            try:
                pinecone_filter = self.build_pinecone_filter(filters)
                query_embedding = self.embed_text(query) if query.strip() else self.embed_text(" ".join(search_terms or []))
                
                if pinecone_filter:
                    logger.info(f"Trying filter search with: {pinecone_filter}")
                    results = self.index.query(
                        vector=query_embedding,
                        filter=pinecone_filter,
                        top_k=top_k,
                        include_metadata=True,
                        namespace=namespace
                    ).matches
                    
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
                results = self.vector_search(search_query, namespace, top_k=top_k)
                
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
        formatted_context = self.format_abstracts_for_medical_prompt(retrieved_docs)
        
        try:
            # Use the custom medical expert prompt
            formatted_prompt = PROMPT.format(
                context=formatted_context,
                question=query+' with trial-ids.'
            )
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert medical orator analyzing conference abstracts. Follow the given prompt template exactly."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.1  # Lower temperature for more consistent medical responses
            )
            
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
        acronym = request.acronym
        request.conferenceName, request.conferenceIteration = extract_acronym_data(acronym)
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

# Hybrid RAG query endpoint
@app.post("/api/conference/query_hybrid", response_model=QueryResponseFilter, dependencies=[Depends(verify_api_key)])
async def query_conference_data_hybrid(
    request: ConferenceQueryRequest,
    rag: HybridRAG = Depends(get_hybrid_rag)
):
    """
    Query conference data using hybrid RAG approach with advanced filtering and classification.
    """
    try:
        # Extract conference information
        acronym = request.acronym
        request.conferenceName, request.conferenceIteration = extract_acronym_data(acronym)
        
        # Create namespace
        namespace = f"{request.conferenceName.lower()}_{request.conferenceIteration}"
        
        logger.info(f"Processing hybrid query: '{request.query}' for {request.conferenceName} {request.conferenceIteration}")
        
        # Process query using hybrid RAG
        result = await rag.process_query(
            query=request.query,
            namespace=namespace,
            save_result=request.save_result,
            top_k=request.top_k
        )
        
        # Convert to response model
        response = QueryResponseFilter(
            query_type=result["query_type"],
            explanation=result["explanation"],
            response=result["response"],
            original_query=result["original_query"],
            filters_applied=result.get("filters_applied"),
            search_terms=result.get("search_terms"),
            is_counting_query=result.get("is_counting_query"),
            expected_fields=result.get("expected_fields"),
            total_results_found=result.get("total_results_found"),
            processing_time_ms=result.get("processing_time_ms"),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Hybrid query processed successfully: {request.query[:50]}...")
        return response
        
    except Exception as e:
        logger.error(f"Error in hybrid query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Job status endpoint (existing functionality)
@app.get("/api/job_status/{job_id}", dependencies=[Depends(verify_api_key)])
async def get_job_status(job_id: str):
    """Get the status of a background job"""
    if job_id not in background_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = background_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job_info.get("status", "unknown"),
        "processed_items": job_info.get("processed_items", 0),
        "total_items": job_info.get("total_items", 0),
        "current_page": job_info.get("current_page", 0),
        "total_pages": job_info.get("total_pages", 0),
        "failed_pages": job_info.get("failed_pages", 0),
        "message": job_info.get("message")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8150)