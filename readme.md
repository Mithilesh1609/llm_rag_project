# LLM Project

This repository contains two main components for language model processing:

## 1. Embedding Generation Pipeline

The `embedding_generation_pipeline` handles the creation and storage of vector embeddings from text data. This pipeline is responsible for:

- Processing input text documents
- Generating vector embeddings using a language model
- Storing these embeddings for later retrieval and comparison

## 2. Query Testing Pipeline

The `query_testing_pipeline` allows for testing and evaluating queries against the generated embeddings. This pipeline:

- Accepts user queries as input
- Converts queries to embeddings in the same vector space
- Performs similarity matching against the stored embeddings
- Returns relevant results based on semantic similarity

## Getting Started

### Prerequisites

- Python 3.11

### Installation

```bash
# Clone the repository
git clone https://github.com/mithilesh1609/llm_project
cd llm_project

# Install dependencies
pip install -r requirements.txt
```

### Usage older way

#### Embedding Generation

```bash
python embedding_generation_pipeline.py
```

#### Query Testing

```bash
python query_testing_pipeline.py
```

### new way
go to production_setup folder and then run this command
``` bash
uvicorn main:app --host 0.0.0.0 --port 8070 --reload --workers 4
```
