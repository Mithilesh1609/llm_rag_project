# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone as LangchainPinecone
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# import os

# # Initialize Pinecone
# pc = Pinecone(os.environ.get("PINECONE_API_KEY"))
# openai_api_key = os.environ.get("OPENAI_API_KEY")
# index_name = "openai-large-embed-v5-full"

# # Connect to existing index
# index = pc.Index(index_name)

# # Initialize Embeddings Model
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

# # Load vector store
# # vectorstore = LangchainPinecone(index, embedding_model,"page_content")
# # Load vector store - CORRECTED VERSION
# vectorstore = PineconeVectorStore(
#     index=index,
#     embedding=embedding_model,
#     text_key="page_content"
# )

# # Create Retriever
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# # Initialize LLM (Optional: For generating better answers)
# llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key,temperature=0)

# # Define a query
# query_text = "What were the key findings from the study on daratumumab monotherapy for smoldering multiple myeloma?"
# # query_text = "What time does the poster session on platelets and megakaryocytes take place?"

# prompt_template = """ You are a expert medical orator which analysis abstracts of the given medical conference and tries to answer the user question.
# If the abstracts contain ANY information related to the question, even if incomplete, summarize what is available, and always stick to the given abstract do not provide any other information from your own.
# Only say you cannot answer if there is absolutely no relevant information, try to give answer in bullet points which is easier to understand and try to go deep in the abstract and identify all necessary information for the given query and summarize well.
# QUESTION: {question}

# RETRIEVED ABSTRACTS:
# {context}

# ANSWER:
# """

# PROMPT = PromptTemplate(
#     template=prompt_template,
#     input_variables=["context", "question"]
# )

# # Create enhanced QA chain with simpler parameters
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": PROMPT}
# )

# # Execute the query
# result = qa_chain({"query": query_text})

# # Print the answer
# print("\n🔍 Answer:")
# if isinstance(result, dict) and "result" in result:
#     print(result["result"])
#     if "source_documents" in result:
#         print("\n📚 Source Documents Used:")
#         for i, doc in enumerate(result["source_documents"]):
#             print(f"\n--- Source {i+1} ---")
#             print(f"Content: {doc.page_content}")
#             print(f"Metadata: {doc.metadata}")
# else:
#     print(result) 

from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
from langchain.chat_models import ChatOpenAI
import os
import json
from datetime import datetime

# Initialize Pinecone
pc = Pinecone(os.environ.get("PINECONE_API_KEY"))
openai_api_key = os.environ.get("OPENAI_API_KEY")
index_name = "openai-large-embed-v5-full"

# Connect to existing index
index = pc.Index(index_name)

# Initialize Embeddings Model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

# Load vector store
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    text_key="page_content"
)

# Create Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key, temperature=0)

# Define prompt template
prompt_template = """You are a expert medical orator which analysis abstracts of the given medical conference and tries to answer the user question.
If the abstracts contain ANY information related to the question, even if incomplete, summarize what is available, and always stick to the given abstract do not provide any other information from your own.
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

# List of queries
query_list = [
    "What were the key findings from the Phase 3 study on daratumumab monotherapy for smoldering multiple myeloma?",
    "How does CD30-directed CAR-T therapy perform in early-phase trials for relapsed/refractory Hodgkin lymphoma and CD30+ lymphomas?",
    "What are the latest industry-sponsored developments in multiple myeloma therapy?",
    "What are the latest insights into mitochondrial metabolism in hematologic malignancies?",
    "How do novel factor VIII replacement therapies impact bleeding control in hemophilia A?",
    "What time does the poster session on platelets and megakaryocytes take place?"
]

# Create log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"query_results_{timestamp}.txt"
json_filename = f"query_results_{timestamp}.json"

# Store results in a structured format for JSON
all_results = []

# Process each query
with open(log_filename, "w") as log_file:
    for i, query in enumerate(query_list):
        print(f"\n\nProcessing Query {i+1}/{len(query_list)}: {query}")
        log_file.write(f"\n\n{'='*80}\nQUERY {i+1}: {query}\n{'='*80}\n")
        
        # Execute the query
        try:
            result = qa_chain({"query": query})
            
            # Print and log the answer
            answer = result["result"]
            print(f"\n🔍 Answer:")
            print(answer)
            log_file.write(f"\nANSWER:\n{answer}\n\n")
            
            # Print and log source documents
            print(f"\n📚 Source Documents Used:")
            log_file.write(f"SOURCE DOCUMENTS:\n\n")
            
            source_docs = []
            for j, doc in enumerate(result["source_documents"]):
                source_info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                source_docs.append(source_info)
                
                # Print to console
                print(f"\n--- Source {j+1} ---")
                print(f"Content: {doc.page_content}")
                print(f"Metadata: {doc.metadata}")
                
                # Write to log file
                log_file.write(f"--- SOURCE {j+1} ---\n")
                log_file.write(f"Content: {doc.page_content}\n")
                log_file.write(f"Metadata: {str(doc.metadata)}\n\n")
            
            # Add to structured results
            all_results.append({
                "query": query,
                "answer": answer,
                "sources": source_docs
            })
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"\n❌ {error_msg}")
            log_file.write(f"\nERROR: {error_msg}\n")
            all_results.append({
                "query": query,
                "error": error_msg,
                "answer": None,
                "sources": []
            })

# Save JSON results
with open(json_filename, "w") as json_file:
    json.dump(all_results, json_file, indent=2, default=str)

print(f"\nResults saved to {log_filename} and {json_filename}")