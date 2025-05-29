from functools import lru_cache
from typing import Annotated, Optional

from fastapi import Depends
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: Optional[str] = Field(None)
    model_name: Optional[str] = Field(None)
    embedding_model_name: Optional[str] = Field(None)
    
    GRAPH_DB_PROVIDER: str = Field("neo4j", description="The graph database provider to use ('neo4j' or 'kuzudb')")
    
    # Neo4j settings (optional based on provider)
    neo4j_uri: Optional[str] = Field(None, description="URI for Neo4j database")
    neo4j_user: Optional[str] = Field(None, description="Username for Neo4j database")
    neo4j_password: Optional[str] = Field(None, description="Password for Neo4j database")

    # KuzuDB settings (optional based on provider)
    KUZUDB_DATABASE_PATH: Optional[str] = Field(None, description="Path to the KuzuDB database file. Required if provider is 'kuzudb'.")
    KUZUDB_IN_MEMORY: bool = Field(False, description="Run KuzuDB in in-memory mode.")

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    @model_validator(mode='after')
    def check_provider_settings(cls, values):
        provider = values.get('GRAPH_DB_PROVIDER')
        if provider == "neo4j":
            if not all([values.get('neo4j_uri'), values.get('neo4j_user'), values.get('neo4j_password')]):
                raise ValueError("For 'neo4j' provider, NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set.")
        elif provider == "kuzudb":
            if not values.get('KUZUDB_IN_MEMORY') and not values.get('KUZUDB_DATABASE_PATH'):
                raise ValueError("For 'kuzudb' provider, KUZUDB_DATABASE_PATH must be set unless KUZUDB_IN_MEMORY is true.")
            if values.get('KUZUDB_IN_MEMORY') and values.get('KUZUDB_DATABASE_PATH'):
                # Allowing path for in-memory to potentially load initial data, though Kuzu usually expects empty path for pure in-memory.
                # Depending on KuzuDB client behavior, this might need adjustment. For now, allow both.
                pass
        else:
            raise ValueError(f"Unsupported GRAPH_DB_PROVIDER: {provider}. Must be 'neo4j' or 'kuzudb'.")
        return values


@lru_cache
def get_settings():
    return Settings()


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
