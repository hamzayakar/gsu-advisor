from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # LLM APIs
    gemini_api_key: str | None = Field(default=None, env="GEMINI_API_KEY")
    
    # Kloudeks Configs
    kloudeks_api_key: str | None = Field(default=None, env="KLOUDEKS_API_KEY")
    kloudeks_base_url: str = Field(default=..., env="KLOUDEKS_BASE_URL")
    
    # Model Names (Defaults can be overridden via .env)
    embedding_model: str = Field(default="qwen3-embedding-8b", env="EMBEDDING_MODEL")
    agent_model: str = Field(default="gpt-oss-120b", env="AGENT_MODEL")
    router_model: str = Field(default="qwen3.5-35b", env="ROUTER_MODEL")
    parser_model: str = Field(default="qwen3.5-35b", env="PARSER_MODEL")
    
    # Qdrant Vector DB
    qdrant_url: str = Field(default=..., env="QDRANT_URL")
    qdrant_api_key: str = Field(default=..., env="QDRANT_API_KEY")
    
    # Langfuse (Optional)
    langfuse_public_key: str | None = Field(default=None, env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, env="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", env="LANGFUSE_HOST")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" 
    )

settings = Settings()