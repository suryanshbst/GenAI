import os
import time  # <-- Added for the hard pause
from pathlib import Path
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = Groq(api_key=groq_api_key)
pdf_path = Path(__file__).parent / "nodejs.pdf"

# 1. Load PDF
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
split_docs = text_splitter.split_documents(documents=docs)
print(f"Total chunks created from PDF: {len(split_docs)}")

# 3. Create embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=gemini_api_key,
)

# 4. Initialize an empty Qdrant Vector Store instance first
vector_store = QdrantVectorStore.from_texts(
    texts=["Initialization placeholder text"], 
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_langchain",
)

# 5. Process and upload chunks in small, paced batches manually
# Adjust batch_size lower if you still hit limits. 15-20 is very safe.
# 5. Process and upload chunks in small, paced batches manually
# Lowering this ensures we never exceed 100 embedded items per minute.
BATCH_SIZE = 5  
print(f"Indexing documents in small batches of {BATCH_SIZE} with pauses to safely stay under the 100/min quota...")

for i in range(0, len(split_docs), BATCH_SIZE):
    batch = split_docs[i : i + BATCH_SIZE]
    
    # Extract texts and metadatas from the document batch
    texts = [doc.page_content for doc in batch]
    metadatas = [doc.metadata for doc in batch]
    
    # Add this chunk batch to Qdrant
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    print(f"✅ Indexed chunks {i} to {min(i + BATCH_SIZE, len(split_docs))}")
    
    # Take a mandatory pause between API batches
    if i + BATCH_SIZE < len(split_docs):
        print("⏳ Sleeping for 4 seconds to safeguard Gemini free tier quota...")
        time.sleep(4)

print("🎉 Document indexing completed successfully!")

# 6. Take user query
query = input("\n> ")

# 7. Retrieve relevant chunks
relevant_chunks = vector_store.similarity_search(
    query=query,
    k=4,
)

# 8. Build context
context = "\n\n".join(
    [
        f"""Page Number: {doc.metadata.get("page_label", doc.metadata.get("page", 0) + 1)}

Content:
{doc.page_content}
"""
        for doc in relevant_chunks
    ]
)

# 9. Create system prompt
SYSTEM_PROMPT = f"""
You are a helpful AI assistant.
Answer the user's question only using the provided context.
If the answer is not available in the context, respond with:
"I couldn't find the answer in the provided PDF."
Whenever possible, mention the page number where the information was found.
Context:
{context}
"""

# 10. Generate response via Groq
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": query,
        },
    ],
)

# 11. Print answer
print("\n🤖 Answer:\n")
print(response.choices[0].message.content)