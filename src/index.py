from pymilvus import MilvusClient, DataType
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load env
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY not set")

# 1️⃣ Load PDF
loader = PyPDFLoader("attention.pdf")
documents = loader.load()

# 2️⃣ Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

texts = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]

# 3️⃣ Create embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

embeddings = embedding_model.embed_documents(texts)
embedding_dim = len(embeddings[0])

# 4️⃣ Connect to Milvus
client = MilvusClient(
    uri="http://localhost:19530"
)

COLLECTION_NAME = "attention_pdf"

# 5️⃣ Create collection (if not exists)
if not client.has_collection(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=embedding_dim,
        metric_type="COSINE",
        consistency_level="Strong",
    )

# 6️⃣ Insert data
data = [
    {
        "id": i,
        "vector": embeddings[i],
        "text": texts[i],
        "metadata": metadatas[i],
    }
    for i in range(len(texts))
]

client.insert(
    collection_name=COLLECTION_NAME,
    data=data
)

print("✅ PDF embeddings stored in Milvus")