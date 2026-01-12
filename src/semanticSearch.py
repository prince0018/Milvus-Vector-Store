from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient, DataType
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)
query = "What is attention mechanism?"
query_embedding = embedding_model.embed_query(query)
client = MilvusClient(
    uri="http://localhost:19530"
)

results = client.search(
    collection_name="attention_pdf",
    data=[query_embedding],
    limit=3,
    output_fields=["text", "metadata"]
)

for hit in results[0]:
    print("\n----------------")
    print("Score:", hit["distance"])
    print(hit["text"][:400])