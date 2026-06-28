import os
from groq import Groq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# We only need the embedding model to embed the single user question
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# Connect instantly to the existing storage on your computer
vector_store = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedding_model,
)

print("⚡ Connected to Qdrant vector database. Ask your question:")
query = input("\n> ")

# Retrieve relevant pieces instantly
relevant_chunks = vector_store.similarity_search(query=query, k=12)

context = "\n\n".join(
    [
        f"Page Number: {doc.metadata.get('page_label', doc.metadata.get('page', 0) + 1)}\nContent:\n{doc.page_content}"
        for doc in relevant_chunks
    ]
)

SYSTEM_PROMPT = f"""
You are a helpful AI assistant.
Answer the user's question only using the provided context.
If the answer is not available in the context, respond with: "I couldn't find the answer in the provided PDF."
Mention the page number where the information was found.

Context:
{context}
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ],
)

print("\n🤖 Answer:\n")
print(response.choices[0].message.content)