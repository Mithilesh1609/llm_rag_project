# Vector Database API Documentation

This documentation outlines the endpoints available in the Vector Database API, which is designed for managing document embeddings in Pinecone, particularly for conference data.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Endpoints](#endpoints)
   - [Ingest Conference Data](#ingest-conference-data)
   - [Update Conference Data](#update-conference-data)
   - [Job Status](#job-status)
   - [Query Embeddings](#query-embeddings)
   - [Conference Query (Streaming)](#conference-query-streaming)
   - [Delete Embeddings](#delete-embeddings)

## Environment Setup

The API requires the following environment variables to be set:

- `PINECONE_API_KEY`: Your Pinecone API key
- `OPENAI_API_KEY`: Your OpenAI API key
- `LARVOL_API_KEY`: API key for the Larvol conference data service

## Endpoints



**Work Done:**


### Ingest Conference Data

**POST** `/api/ingest_conference_data`

Fetches and processes conference data from the Larvol API as a background job.

**Request Body:**
```json
{
  "conferenceName": "string",
  "conferenceIteration": "string",
  "max_pages": 80 (optional)
}
```

**Response:**
```json
{
  "success": true,
  "message": "Started ingestion of conference data for [conferenceName] [conferenceIteration]",
  "document_ids": [],
  "count": 0,
  "job_id": "job_id_string"
}
```

**Work Done:**
- Starts a background job that fetches conference data from the Larvol API
- Creates a Pinecone index if it doesn't exist
- Splits documents into chunks using RecursiveCharacterTextSplitter
- Generates embeddings using OpenAI's text-embedding-3-large model
- Stores embeddings and metadata in Pinecone
- Job progress can be tracked using the returned job_id

### Update Conference Data

**POST** `/api/update_conference_data`

Updates conference data embeddings, but only if the `Conf_Upload_Version` has changed.

**Request Body:**
```json
{
  "conferenceName": "string",
  "conferenceIteration": "string",
  "max_pages": 80 (optional)
}
```

**Response:**
```json
{
  "success": true,
  "message": "Started updating conference data for [conferenceName] [conferenceIteration]",
  "document_ids": [],
  "count": 0,
  "job_id": "job_id_string"
}
```

**Work Done:**
- Fetches the latest conference data from the Larvol API
- Checks each document's `Conf_Upload_Version` against what's already in the database
- Only updates documents that have a new version
- Deletes old vectors and creates new ones for updated documents
- Tracks statistics for updated vs. skipped documents

### Job Status

**GET** `/api/job_status/{job_id}`

Gets the status of a background job for ingesting or updating conference data.

**Path Parameters:**
- `job_id`: The job ID returned from the ingest or update endpoint

**Response:**
```json
{
  "job_id": "string",
  "status": "string",
  "processed_items": 0,
  "total_items": 0,
  "current_page": 0,
  "total_pages": 0,
  "message": "string"
}
```

**Work Done:**
- Retrieves the current status of the specified background job
- For update jobs, also reports updated_items and skipped_items counts

### Query Embeddings

**POST** `/api/query`

Queries Pinecone for similar documents based on a text query.

**Request Body:**
```json
{
  "conferenceName": "string",
  "conferenceIteration": "string",
  "query_text": "string",
  "top_k": 5, (optional)
  "include_metadata": true,(optional)
  "filter": {}(optional)
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "string",
      "score": 0.0,
      "metadata": {},
      "content": "string"
    }
  ],
  "count": 0,
  "query_time_ms": 0.0
}
```

**Work Done:**
- Generates an embedding for the query text
- Searches the Pinecone index for semantically similar vectors
- Returns the top k results with their similarity scores and metadata

### Conference Query (Streaming)

**POST** `/api/conference/query_streaming`

Queries conference data and optionally uses RAG to generate a natural language response with streaming capability.

**Request Body:**
```json
{
  "conferenceName": "string",
  "conferenceIteration": "string",
  "query": "string",
  "top_k": 5,(optional)
  "use_rag": true,(optional)
  "model_name": "gpt-4o-mini",(optional)
  "temperature": 0.0 (optional)
}
```

**Response:**
Streams a plain markdown response when use_rag is true, or returns a JSON response:

```json
{
  "query": "string",
  "answer": "string",
  "sources": [
    {
      "content": "string",
      "metadata": {}
    }
  ],
  "retrieval_time_ms": 0.0,
  "generation_time_ms": 0.0,
  "total_time_ms": 0.0
}
```

**Work Done:**
- For RAG mode: 
  - Retrieves relevant documents from Pinecone
  - Uses a large language model to generate a response based on the retrieved content
  - Streams the response back in real-time
- For vector search only mode: 
  - Performs a vector search and returns the results without LLM processing

### Delete Embeddings

**POST** `/api/delete_embeddings`

Deletes embeddings from Pinecone by ID.

**Request Body:**
```json
{
  "conferenceName": "string",
  "conferenceIteration": "string",
  "ids": ["id1", "id2", "..."],
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully deleted X embeddings",
  "document_ids": [],
  "count": 0,
  "job_id": null
}
```

**Work Done:**
- Deletes the specified vectors from the Pinecone index by their IDs