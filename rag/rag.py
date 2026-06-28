import os
from pathlib import Path
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
# from langchain_openai import OpenAIEmbeddings

groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = Groq(
    api_key=groq_api_key,
)
pdf_path = Path(__file__).parent / "nodejs.pdf"

# Load PDF
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
split_docs = text_splitter.split_documents(documents=docs)

# Create embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=gemini_api_key,
)

"""
OpenAI Embeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY"),
)
"""

# Create vector store and index documents
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedding_model,
    batch_size=16
)
print("✅ Document indexing completed.")

# Connect to existing collection
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedding_model,
)

# Take user query
query = input("\n> ")

# Retrieve relevant chunks
relevant_chunks = retriever.similarity_search(
    query=query,
    k=4,
)

# Build context
context = "\n\n".join(
    [
        f"""Page Number: {doc.metadata.get("page_label", doc.metadata.get("page", 0) + 1)}

Content:
{doc.page_content}
"""
        for doc in relevant_chunks
    ]
)

# Create system prompt
SYSTEM_PROMPT = f"""
You are a helpful AI assistant.
Answer the user's question only using the provided context.
If the answer is not available in the context, respond with:
"I couldn't find the answer in the provided PDF."
Whenever possible, mention the page number where the information was found.
Context:
{context}
"""

# Generate response
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

# Print answer
print("\n🤖 Answer:\n")
print(response.choices[0].message.content)