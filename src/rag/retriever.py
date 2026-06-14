import logging
from openai import OpenAI
from qdrant_client import QdrantClient
from src.config.settings import settings

logger = logging.getLogger(__name__)

class GSURetriever:
    def __init__(self, collection_name: str = "gsu_rules"):
        self.llm_client = OpenAI(
            base_url=settings.kloudeks_base_url,
            api_key=settings.kloudeks_api_key
        )
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = collection_name
        self.embedding_model = settings.embedding_model

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Convert the user query into a vector, search in Qdrant, and return the most relevant rules."""
        logger.info(f"Retrieving top {top_k} rules for query: '{query}'")
        
        try:
            # 1. Convert user query to vector
            response = self.llm_client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            query_vector = response.data[0].embedding
            
            # 2. Search in Qdrant
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=0.45
            ).points
            
            # 3. Format results for the agent to read
            results = []
            for hit in search_result:
                results.append({
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "topic": hit.payload.get("topic", "Unknown")
                })
                
            return results
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []