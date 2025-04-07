import json,os
import uuid
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
pc = Pinecone(pinecone_api_key)
json_data = load_json("/home/learnsense/personal/freelance/llm_rag_project/json_data/ASH_Conf_2024_sampled_final.json")
# json_data = load_json("/home/learnsense/personal/freelance/llm_rag_project/json_data/sample_non_author_data_v2.json")
# index_name = "llama-embed-v2-non-author-with-time-v2"
# index_name = "openai-large-embed-v5-full"
index_name = "llama-text-embed-1024-v1"

# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=3072,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region='us-east-1')
#     )

index = pc.Index(index_name)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300) #1800,250

documents = []
for item in json_data['ASH_Conf_2024']:
    if(item!='ASH_Conf_2024'):
        # content = f"Title: {item['article_title']}\nSession: {item['session_title']}\nAbstract: {item['abstract_text']}\nCategory: {item['category']}"
        content = f"Title: {item.get('article_title', 'N/A')}\n" \
                f"Session: {item.get('session_title', 'N/A')}\n" \
                f"Abstract: {item.get('abstract_text', 'N/A')}\n" \
                f"Category: {item.get('category', 'N/A')}\n" \
                f"session Date: {item.get('date', 'N/A')}\n" \
                f"Location: {item.get('location', 'N/A')}\n" \
                f"Start Time: {item.get('start_time', 'N/A')}\n"\
                f"End Time: {item.get('end_time', 'N/A')}\n"
        # content = f"Abstract: {item.get('abstract_text', 'N/A')}\n"
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            metadata = {
                "manual_id": item.get("manual_id", ""),
                "source_id": item.get("source_id", ""),
                "url": item.get("url", ""),
                "date": item.get("date", ""),
                "location": item.get("location", ""),
                "category": item.get("category", ""),
                "article_title": item.get("article_title", ""),
                "session_title": item.get("session_title", ""),
                "start_time": item.get("start_time", ""),
                "end_time": item.get("end_time", ""),
            }
            documents.append(Document(page_content=chunk, metadata=metadata,id=uuid.uuid4()))
            
#### openai embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

# # 3. Embedding and Upserting
batch_size = 100  # Adjust batch size as needed
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i + batch_size]
    batch_texts = [doc.page_content for doc in batch_docs]
    print(f"Processing documents {i} to {i + len(batch_texts)}")
    embeddings = embedding_model.embed_documents(batch_texts)
    vectors = []

    for j, embedding_values in enumerate(embeddings):
    # Include the page_content in the metadata
        metadata = batch_docs[j].metadata.copy()
        metadata["page_content"] = batch_docs[j].page_content
        vectors.append((str(i + j), embedding_values, metadata))
    index.upsert(vectors=vectors)
   
########## llama embeddings
# Import other necessary libraries

# Replace OpenAI embeddings with Llama embeddings

# The rest of your code remains the same
batch_size = 50  # Adjust batch size as needed
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i + batch_size]
    batch_texts = [doc.page_content for doc in batch_docs]
    print(f"Processing documents {i} to {i + len(batch_texts)}")
    embeddings = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=batch_texts,
        parameters={"input_type": "passage","truncate": "END","dimension": 2048},
    )
    vectors = []
    
    # for j, embedding_values in enumerate(embeddings):
    #     # Include the page_content in the metadata
    #     metadata = batch_docs[j].metadata.copy()
    #     metadata["page_content"] = batch_docs[j].page_content
    #     vectors.append((str(i + j), embedding_values, metadata))
    
    vectors = []
    for d, e in zip(batch_docs, embeddings):
        vectors.append({
            "id": d.id,
            "values": e['values'],
            "metadata": {'text': str(d.metadata)}
        })

    index.upsert(
        vectors=vectors,
        namespace="ns1"
    )

    # index.upsert(vectors=vectors)