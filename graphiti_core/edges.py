"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from time import time
from typing import Any
from uuid import uuid4

# from neo4j import AsyncDriver # No longer directly used by Edge classes
from pydantic import BaseModel, Field
from typing_extensions import LiteralString

from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError
# from graphiti_core.helpers import DEFAULT_DATABASE, parse_db_date # No longer directly used here
from graphiti_core.helpers import parse_db_date # parse_db_date is still used by get_entity_edge_from_record
from graphiti_core.providers.base import GraphDatabaseProvider # Import Provider
from graphiti_core.models.edges.edge_db_queries import ( # These queries are used by Neo4jProvider, not directly here
    COMMUNITY_EDGE_SAVE,
    ENTITY_EDGE_SAVE,
    EPISODIC_EDGE_SAVE,
)
from graphiti_core.nodes import Node # Used in __eq__

logger = logging.getLogger(__name__)

ENTITY_EDGE_RETURN: LiteralString = """
        RETURN
            e.uuid AS uuid,
            startNode(e).uuid AS source_node_uuid,
            endNode(e).uuid AS target_node_uuid,
            e.created_at AS created_at,
            e.name AS name,
            e.group_id AS group_id,
            e.fact AS fact,
            e.episodes AS episodes,
            e.expired_at AS expired_at,
            e.valid_at AS valid_at,
            e.invalid_at AS invalid_at,
            properties(e) AS attributes
            """


class Edge(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    group_id: str = Field(description='partition of the graph')
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime

    @abstractmethod
    async def save(self, provider: GraphDatabaseProvider):
        """Saves the current edge instance to the database using the provided provider."""
        pass

    async def delete(self, provider: GraphDatabaseProvider):
        """Deletes the current edge instance from the database using the provided provider."""
        await provider.delete_edge(self.uuid) # Delegate to provider
        logger.debug(f'Deleted Edge: {self.uuid} via provider')
        return None

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Edge): # Changed Node to Edge for more accurate comparison
            return self.uuid == other.uuid
        return False

    @classmethod
    async def get_by_uuid(cls, provider: GraphDatabaseProvider, uuid: str):
        """Retrieves an edge of this type by its UUID using the provided provider."""
        pass


class EpisodicEdge(Edge):
    async def save(self, provider: GraphDatabaseProvider):
        """Saves the current EpisodicEdge instance using the provided provider."""
        logger.debug(f'Saving EpisodicEdge: {self.uuid} via provider')
        return await provider.save_episodic_edge(self)

    @classmethod
    async def get_by_uuid(cls, provider: GraphDatabaseProvider, uuid: str) -> "EpisodicEdge":
        """Retrieves an EpisodicEdge by its UUID using the provided provider."""
        edge = await provider.get_episodic_edge_by_uuid(uuid)
        if not edge:
            raise EdgeNotFoundError(uuid)
        return edge

    @classmethod
    async def get_by_uuids(cls, provider: GraphDatabaseProvider, uuids: list[str]) -> List["EpisodicEdge"]:
        """Retrieves multiple EpisodicEdges by their UUIDs using the provided provider."""
        return await provider.get_episodic_edges_by_uuids(uuids)

    @classmethod
    async def get_by_group_ids(
        cls,
        provider: GraphDatabaseProvider,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> List["EpisodicEdge"]:
        """Retrieves EpisodicEdges by group IDs using the provided provider."""
        return await provider.get_episodic_edges_by_group_ids(
            group_ids=group_ids, limit=limit, uuid_cursor=uuid_cursor
        )


class EntityEdge(Edge):
    name: str = Field(description='name of the edge, relation name')
    fact: str = Field(description='fact representing the edge and nodes that it connects')
    fact_embedding: list[float] | None = Field(default=None, description='embedding of the fact')
    episodes: list[str] = Field(
        default=[],
        description='list of episode ids that reference these entity edges',
    )
    expired_at: datetime | None = Field(
        default=None, description='datetime of when the node was invalidated'
    )
    valid_at: datetime | None = Field(
        default=None, description='datetime of when the fact became true'
    )
    invalid_at: datetime | None = Field(
        default=None, description='datetime of when the fact stopped being true'
    )
    attributes: dict[str, Any] = Field(
        default={}, description='Additional attributes of the edge. Dependent on edge name'
    )

    async def generate_embedding(self, embedder: EmbedderClient):
        start = time()
        text = self.fact.replace('\n', ' ')
        embedding_result = await embedder.create(input_data=[text])
        if embedding_result and len(embedding_result) > 0 and isinstance(embedding_result[0], list):
             self.fact_embedding = embedding_result[0] # Assuming create returns List[List[float]] for single input
        elif embedding_result and isinstance(embedding_result, list) and len(embedding_result) > 0 and isinstance(embedding_result[0], float):
             self.fact_embedding = embedding_result # If it's already List[float]
        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')
        return self.fact_embedding

    async def load_fact_embedding(self, provider: GraphDatabaseProvider): # Changed driver to provider
        edge_data = await provider.get_entity_edge_by_uuid(self.uuid)
        if edge_data and edge_data.fact_embedding:
            self.fact_embedding = edge_data.fact_embedding
        else:
            logger.warning(f"Could not load fact_embedding for EntityEdge {self.uuid} via provider.")
            if not edge_data: # Raise if edge itself was not found
                raise EdgeNotFoundError(self.uuid)


    async def save(self, provider: GraphDatabaseProvider):
        """Saves the current EntityEdge instance using the provided provider."""
        logger.debug(f'Saving EntityEdge: {self.uuid} via provider')
        return await provider.save_entity_edge(self)

    @classmethod
    async def get_by_uuid(cls, provider: GraphDatabaseProvider, uuid: str) -> "EntityEdge":
        """Retrieves an EntityEdge by its UUID using the provided provider."""
        edge = await provider.get_entity_edge_by_uuid(uuid)
        if not edge:
            raise EdgeNotFoundError(uuid)
        return edge

    @classmethod
    async def get_by_uuids(cls, provider: GraphDatabaseProvider, uuids: list[str]) -> List["EntityEdge"]:
        """Retrieves multiple EntityEdges by their UUIDs using the provided provider."""
        if not uuids: 
            return []
        return await provider.get_entity_edges_by_uuids(uuids)

    @classmethod
    async def get_by_group_ids(
        cls,
        provider: GraphDatabaseProvider,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> List["EntityEdge"]:
        """Retrieves EntityEdges by group IDs using the provided provider."""
        return await provider.get_entity_edges_by_group_ids(
            group_ids=group_ids, limit=limit, uuid_cursor=uuid_cursor
        )

    @classmethod
    async def get_by_node_uuid(cls, provider: GraphDatabaseProvider, node_uuid: str) -> List["EntityEdge"]:
        """Retrieves EntityEdges connected to a given node UUID using the provided provider."""
        return await provider.get_entity_edges_by_node_uuid(node_uuid)


class CommunityEdge(Edge):
    async def save(self, provider: GraphDatabaseProvider):
        """Saves the current CommunityEdge instance using the provided provider."""
        logger.debug(f'Saving CommunityEdge: {self.uuid} via provider')
        return await provider.save_community_edge(self)

    @classmethod
    async def get_by_uuid(cls, provider: GraphDatabaseProvider, uuid: str) -> "CommunityEdge":
        """Retrieves a CommunityEdge by its UUID using the provided provider."""
        edge = await provider.get_community_edge_by_uuid(uuid)
        if not edge: 
            raise EdgeNotFoundError(uuid)
        return edge

    @classmethod
    async def get_by_uuids(cls, provider: GraphDatabaseProvider, uuids: list[str]) -> List["CommunityEdge"]:
        """Retrieves multiple CommunityEdges by their UUIDs using the provided provider."""
        return await provider.get_community_edges_by_uuids(uuids)

    @classmethod
    async def get_by_group_ids(
        cls,
        provider: GraphDatabaseProvider,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> List["CommunityEdge"]:
        """Retrieves CommunityEdges by group IDs using the provided provider."""
        return await provider.get_community_edges_by_group_ids(
            group_ids=group_ids, limit=limit, uuid_cursor=uuid_cursor
        )


# Edge helpers
# These helpers are primarily used by Neo4jProvider for parsing raw query results.
# Other providers (like KuzuDBProvider) might have their own internal parsing methods
# that directly construct Pydantic models.
def get_episodic_edge_from_record(record: Any) -> EpisodicEdge:
    return EpisodicEdge(
        uuid=record['uuid'],
        group_id=record.get('group_id'), # Use .get for safety
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=record['created_at'].to_native() if hasattr(record['created_at'], 'to_native') else record['created_at'],
    )


def get_entity_edge_from_record(record: Any) -> EntityEdge:
    # Similar to get_entity_node_from_record, this helper might need to be provider-aware
    # or simplified if providers return Pydantic models directly.
    # Assuming Neo4j-like record structure for now.
    attributes = record.get('attributes', {})

    edge = EntityEdge(
        uuid=record['uuid'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        fact=record['fact'],
        name=record['name'],
        group_id=record.get('group_id'),
        episodes=record.get('episodes', []),
        created_at=record['created_at'].to_native() if hasattr(record['created_at'], 'to_native') else record['created_at'],
        expired_at=parse_db_date(record.get('expired_at')),
        valid_at=parse_db_date(record.get('valid_at')),
        invalid_at=parse_db_date(record.get('invalid_at')),
        attributes=attributes,
        fact_embedding=record.get('fact_embedding') # Add fact_embedding
    )
    
    if isinstance(edge.attributes, dict):
        # Clean up attributes that are already top-level fields
        for key_to_pop in [
            'uuid', 'source_node_uuid', 'target_node_uuid', 'fact', 'name', 
            'group_id', 'episodes', 'created_at', 'expired_at', 'valid_at', 
            'invalid_at', 'fact_embedding'
        ]:
            edge.attributes.pop(key_to_pop, None)

    return edge


def get_community_edge_from_record(record: Any) -> CommunityEdge: 
    return CommunityEdge(
        uuid=record['uuid'],
        group_id=record.get('group_id'),
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=record['created_at'].to_native() if hasattr(record['created_at'], 'to_native') else record['created_at'],
    )


async def create_entity_edge_embeddings(embedder: EmbedderClient, edges: list[EntityEdge]):
    # This utility function is primarily called by Graphiti class or other high-level logic
    # before passing edges to a provider's bulk save operation, or for ad-hoc embedding.
    # It operates on Pydantic model instances directly.
    if not edges: # Check if the list is empty
        return
    fact_embeddings_list = await embedder.create_batch([edge.fact for edge in edges])
    for edge, fact_embedding_item in zip(edges, fact_embeddings_list, strict=True):
        # Assuming create_batch returns a list of embeddings (List[List[float]]),
        # and each item in that list is an embedding for the corresponding edge.
        edge.fact_embedding = fact_embedding_item
```
