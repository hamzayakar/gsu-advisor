import json
import uuid
import os
import time
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from src.config.settings import settings

class KloudeksIndexer:
    def __init__(self, collection_name: str = "gsu_rules"):
        self.llm_client = OpenAI(
            base_url=settings.kloudeks_base_url,
            api_key=settings.kloudeks_api_key
        )
        self.embedding_model = settings.embedding_model
        
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=60.0  # Added timeout to prevent read operation errors
        )
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
            )
            print(f"[*] Created Qdrant collection: {self.collection_name}")

    def index_chunks(self, chunks_file_path: str, batch_size: int = 20):
        if not os.path.exists(chunks_file_path):
            print(f"[-] File not found: {chunks_file_path}")
            return

        with open(chunks_file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        print(f"[*] Starting indexing process...")
        print(f"[*] Total {total_chunks} chunks will be processed in {total_batches} batches.")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            points = []
            
            for chunk in batch:
                response = self.llm_client.embeddings.create(
                    model=self.embedding_model,
                    input=chunk["text"]
                )
                vector = response.data[0].embedding
                point_id = str(uuid.uuid4())
                payload = {"text": chunk["text"], **chunk["metadata"]}
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            current_batch = (i // batch_size) + 1
            print(f"  [+] Batch {current_batch}/{total_batches} successfully indexed.")
            
            # Pause to avoid rate limits and allow the server to process
            if current_batch < total_batches:
                time.sleep(2)

        print(f"[+] Successfully indexed all {total_chunks} chunks into Qdrant collection: {self.collection_name}")

if __name__ == "__main__":
    input_file = os.path.join(os.getcwd(), "data", "processed_chunks/chunks.json")
    
    print(f"[*] Starting vector indexing process from: {input_file}")
    
    try:
        indexer = KloudeksIndexer()
        indexer.index_chunks(input_file)
        print("[+] Vector indexing process completed successfully.")
    except Exception as e:
        print(f"[-] An error occurred during vector indexing: {e}")