from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
print("Loading model")
model = SentenceTransformer("google/embeddinggemma-300m")
print("Model loaded")

# Run inference with queries and documents
query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]
print("Encoding query")
query_embeddings = model.encode_query(query)
print("Encoding documents")
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# (768,) (4, 768)

# Compute similarities to determine a ranking
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.3011, 0.6359, 0.4930, 0.4889]])

# Convert similarities to a ranking
ranking = similarities.argsort(descending=True)[0]
print(ranking)
# tensor([1, 2, 3, 0])
