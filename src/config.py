from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = ""
    
    # App Settings
    app_name: str = "SaaS Support RAG API"
    log_level: str = "INFO"
    environment: str = "development"
    
    # Vector DB
    vector_db_path: str = "./data/vectorstore"
    
    # LLM & Embedding Settings
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-3.5-turbo"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
