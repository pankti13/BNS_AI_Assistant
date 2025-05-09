import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

def populate_chroma(collection_name="bns_vectors", dataset_path="data/BNS_Dataset.csv", chroma_path="./chroma_db"):
    model_embed = SentenceTransformer("all-mpnet-base-v2")
    df = pd.read_csv(dataset_path)
    chroma_client = PersistentClient(chroma_path)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    if len(collection.get()['ids']) > 0:
        return collection

    for idx, row in df.iterrows():
        vector = model_embed.encode(row["Section Description"])
        collection.add(
            ids=[f"doc_{idx}"],
            embeddings=[vector],
            metadatas=[{
                "Section Number": row["Section Number"],
                "Chapter Number": row["Chapter Number"],
                "Chapter Name": row["Chapter Name"],
                "Section Title": row["Section Title"],
                "Section Description": row["Section Description"]
            }]
        )
    print ('ChromaDB setup completed.')
    return collection
