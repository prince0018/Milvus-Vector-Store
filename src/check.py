from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

results = client.query(
    collection_name="attention_pdf",
    limit=5,
    output_fields=["text", "metadata"]
)

for i, r in enumerate(results):
    print(f"\n--- Document {i+1} ---")
    print(r["text"][:500])
    print("Metadata:", r["metadata"])