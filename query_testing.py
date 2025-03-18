from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# Initialize Pinecone
pc = Pinecone(os.environ.get("PINECONE_API_KEY"))
openai_api_key = os.environ.get("OPENAI_API_KEY")
index_name = "openai-large-embed-v4"

# Connect to existing index
index = pc.Index(index_name)

# Initialize Embeddings Model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

# Load vector store
vectorstore = LangchainPinecone(index, embedding_model,"page_content")

# Create Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# Initialize LLM (Optional: For generating better answers)
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key,temperature=0)

# Define a query
query_text = "What were the key findings from the study on daratumumab monotherapy for smoldering multiple myeloma?"
# query_text = "What time does the poster session on platelets and megakaryocytes take place?"

prompt_template = """ You are a expert medical orator which analysis abstracts of the given medical conference and tries to answer the user question.
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

# Create enhanced QA chain with simpler parameters
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Execute the query
result = qa_chain({"query": query_text})

# Print the answer
print("\n🔍 Answer:")
if isinstance(result, dict) and "result" in result:
    print(result["result"])
    if "source_documents" in result:
        print("\n📚 Source Documents Used:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"\n--- Source {i+1} ---")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
else:
    print(result) 