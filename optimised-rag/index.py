import os
import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents=docs)
print(f"Total chunks created: {len(split_docs)}")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=gemini_api_key,
)

# Initialize the vector store collection
vector_store = QdrantVectorStore.from_texts(
    texts=["Initialization placeholder"], 
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_langchain",
)

# Slow, paced indexing so you don't hit the free tier quota limits
BATCH_SIZE = 5  
for i in range(0, len(split_docs), BATCH_SIZE):
    batch = split_docs[i : i + BATCH_SIZE]
    texts = [doc.page_content for doc in batch]
    metadatas = [doc.metadata for doc in batch]
    
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    print(f"✅ Indexed chunks {i} to {min(i + BATCH_SIZE, len(split_docs))}")
    if i + BATCH_SIZE < len(split_docs):
        time.sleep(4)

print("🎉 Permanently indexed to local Qdrant collection!")