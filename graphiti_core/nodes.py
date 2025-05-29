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
from enum import Enum
from time import time
from typing import Any
from uuid import uuid4

# from neo4j import AsyncDriver # No longer directly used by Node classes
from pydantic import BaseModel, Field
from typing_extensions import LiteralString

from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import NodeNotFoundError
# from graphiti_core.helpers import DEFAULT_DATABASE # No longer directly used here
from graphiti_core.providers.base import GraphDatabaseProvider # Import Provider
from graphiti_core.models.nodes.node_db_queries import ( # These queries are used by Neo4jProvider, not directly here
    COMMUNITY_NODE_SAVE,
    ENTITY_NODE_SAVE,
    EPISODIC_NODE_SAVE,
)
from graphiti_core.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)

ENTITY_NODE_RETURN: LiteralString = """
        RETURN
            n.uuid As uuid, 
            n.name AS name,
            n.group_id AS group_id,
            n.created_at AS created_at, 
            n.summary AS summary,
            labels(n) AS labels,
            properties(n) AS attributes
            """


class EpisodeType(Enum):
    """
    Enumeration of different types of episodes that can be processed.

    This enum defines the various sources or formats of episodes that the system
    can handle. It's used to categorize and potentially handle different types
    of input data differently.

    Attributes:
    -----------
    message : str
        Represents a standard message-type episode. The content for this type
        should be formatted as "actor: content". For example, "user: Hello, how are you?"
        or "assistant: I'm doing well, thank you for asking."
    json : str
        Represents an episode containing a JSON string object with structured data.
    text : str
        Represents a plain text episode.
    """

    message = 'message'
    json = 'json'
    text = 'text'

    @staticmethod
    def from_str(episode_type: str):
        if episode_type == 'message':
            return EpisodeType.message
        if episode_type == 'json':
            return EpisodeType.json
        if episode_type == 'text':
            return EpisodeType.text
        logger.error(f'Episode type: {episode_type} not implemented')
        raise NotImplementedError


class Node(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description='name of the node')
    group_id: str = Field(description='partition of the graph')
    labels: list[str] = Field(default_factory=list) # Keep for model structure, provider handles storage
    created_at: datetime = Field(default_factory=lambda: utc_now())

    @abstractmethod
    async def save(self, provider: GraphDatabaseProvider): ... # Changed driver to provider

    async def delete(self, provider: GraphDatabaseProvider): # Changed driver to provider
        await provider.delete_node(self.uuid) # Delegate to provider
        logger.debug(f'Deleted Node: {self.uuid} via provider')
        # Return value might change based on provider.delete_node signature, assuming None for now
        return None 

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False

    @classmethod
    async def delete_by_group_id(cls, provider: GraphDatabaseProvider, group_id: str): # Changed driver to provider
        # This method might need to iterate through node types if provider's method is type-specific
        # Or provider.delete_nodes_by_group_id handles it generically (if possible)
        # For now, assuming provider has a method that can delete by group_id across types,
        # or this method needs to be smarter.
        # Let's assume a generic provider method, or this needs to call for each type.
        # Based on current provider methods, it's type specific.
        node_types_to_check = ["Episodic", "Entity", "Community"] # TODO: Make this more robust
        for node_type in node_types_to_check:
            await provider.delete_nodes_by_group_id(group_id=group_id, node_type=node_type)
        logger.info(f"Deletion requested for nodes with group_id: {group_id} across relevant types.")
        return 'SUCCESS'


    @classmethod
    async def get_by_uuid(cls, provider: GraphDatabaseProvider, uuid: str): ... # Changed driver to provider

    @classmethod
    async def get_by_uuids(cls, provider: GraphDatabaseProvider, uuids: list[str]): ... # Changed driver to provider


class EpisodicNode(Node):
    source: EpisodeType = Field(description='source type')
    source_description: str = Field(description='description of the data source')
    content: str = Field(description='raw episode data')
    valid_at: datetime = Field(
        description='datetime of when the original document was created',
    )
    entity_edges: list[str] = Field(
        description='list of entity edges referenced in this episode',
        default_factory=list,
    )

    async def save(self, provider: GraphDatabaseProvider): # Changed driver to provider
        logger.debug(f'Saving EpisodicNode: {self.uuid} via provider')
        return await provider.save_episodic_node(self) # Delegate to provider

    @classmethod
    async def get_by_uuid(cls, provider: GraphDatabaseProvider, uuid: str): # Changed driver to provider
        # Delegate to provider
        node = await provider.get_episodic_node_by_uuid(uuid)
        if not node:
            raise NodeNotFoundError(uuid)
        return node

    @classmethod
    async def get_by_uuids(cls, provider: GraphDatabaseProvider, uuids: list[str]): # Changed driver to provider
        # Delegate to provider
        return await provider.get_episodic_nodes_by_uuids(uuids)

    @classmethod
    async def get_by_group_ids(
        cls,
        provider: GraphDatabaseProvider, # Changed driver to provider
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        # Delegate to provider
        return await provider.get_episodic_nodes_by_group_ids(
            group_ids=group_ids, limit=limit, uuid_cursor=uuid_cursor
        )

    @classmethod
    async def get_by_entity_node_uuid(cls, provider: GraphDatabaseProvider, entity_node_uuid: str): # Changed driver to provider
        # Delegate to provider
        return await provider.get_episodic_nodes_by_entity_node_uuid(entity_node_uuid)


class EntityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.source_description AS source_description,
    summary: str = Field(description='regional summary of surrounding edges', default_factory=str)
    attributes: dict[str, Any] = Field(
        default={}, description='Additional attributes of the node. Dependent on node labels'
    )

    async def generate_name_embedding(self, embedder: EmbedderClient):
        start = time()
        text = self.name.replace('\n', ' ')
        # embedder.create returns List[List[float]]
        embedding_result = await embedder.create(input_data=[text])
        if embedding_result and len(embedding_result) > 0:
            self.name_embedding = embedding_result[0]
        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')
        return self.name_embedding

    async def load_name_embedding(self, provider: GraphDatabaseProvider): # Changed driver to provider
        # Fetch the node using provider to get its embedding
        node_data = await provider.get_entity_node_by_uuid(self.uuid)
        if node_data and node_data.name_embedding:
            self.name_embedding = node_data.name_embedding
        else:
            logger.warning(f"Could not load name_embedding for EntityNode {self.uuid} via provider.")
            # Optionally raise NodeNotFoundError or handle as appropriate
            if not node_data:
                 raise NodeNotFoundError(self.uuid)


    async def save(self, provider: GraphDatabaseProvider): # Changed driver to provider
        logger.debug(f'Saving EntityNode: {self.uuid} via provider')
        return await provider.save_entity_node(self) # Delegate to provider

    @classmethod
    async def get_by_uuid(cls, provider: GraphDatabaseProvider, uuid: str): # Changed driver to provider
        node = await provider.get_entity_node_by_uuid(uuid)
        if not node:
            raise NodeNotFoundError(uuid)
        return node

    @classmethod
    async def get_by_uuids(cls, provider: GraphDatabaseProvider, uuids: list[str]): # Changed driver to provider
        return await provider.get_entity_nodes_by_uuids(uuids)

    @classmethod
    async def get_by_group_ids(
        cls,
        provider: GraphDatabaseProvider, # Changed driver to provider
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        return await provider.get_entity_nodes_by_group_ids(
            group_ids=group_ids, limit=limit, uuid_cursor=uuid_cursor
        )


class CommunityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary: str = Field(description='region summary of member nodes', default_factory=str)

    async def save(self, provider: GraphDatabaseProvider): # Changed driver to provider
        logger.debug(f'Saving CommunityNode: {self.uuid} via provider')
        return await provider.save_community_node(self) # Delegate to provider

    async def generate_name_embedding(self, embedder: EmbedderClient):
        start = time()
        text = self.name.replace('\n', ' ')
        embedding_result = await embedder.create(input_data=[text])
        if embedding_result and len(embedding_result) > 0:
            self.name_embedding = embedding_result[0]
        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')
        return self.name_embedding

    async def load_name_embedding(self, provider: GraphDatabaseProvider): # Changed driver to provider
        node_data = await provider.get_community_node_by_uuid(self.uuid)
        if node_data and node_data.name_embedding:
            self.name_embedding = node_data.name_embedding
        else:
            logger.warning(f"Could not load name_embedding for CommunityNode {self.uuid} via provider.")
            if not node_data:
                raise NodeNotFoundError(self.uuid)


    @classmethod
    async def get_by_uuid(cls, provider: GraphDatabaseProvider, uuid: str): # Changed driver to provider
        node = await provider.get_community_node_by_uuid(uuid)
        if not node:
            raise NodeNotFoundError(uuid)
        return node

    @classmethod
    async def get_by_uuids(cls, provider: GraphDatabaseProvider, uuids: list[str]): # Changed driver to provider
        return await provider.get_community_nodes_by_uuids(uuids)

    @classmethod
    async def get_by_group_ids(
        cls,
        provider: GraphDatabaseProvider, # Changed driver to provider
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        return await provider.get_community_nodes_by_group_ids(
            group_ids=group_ids, limit=limit, uuid_cursor=uuid_cursor
        )


# Node helpers
# These helpers are primarily used by Neo4jProvider now.
# They might be moved to the provider module or a shared parsing utility later.
def get_episodic_node_from_record(record: Any) -> EpisodicNode:
    return EpisodicNode(
        content=record['content'],
        created_at=record['created_at'].to_native().timestamp(), # This might need adjustment based on provider
        valid_at=(record['valid_at'].to_native()),
        uuid=record['uuid'],
        group_id=record['group_id'],
        source=EpisodeType.from_str(record['source']),
        name=record['name'],
        source_description=record['source_description'],
        entity_edges=record['entity_edges'],
    )


def get_entity_node_from_record(record: Any) -> EntityNode:
    # Ensure 'labels' and 'attributes' are present in the record if expected by EntityNode
    # The structure of 'record' depends on what the provider's get_* methods return.
    # Neo4j returns 'labels' and 'attributes' as separate dict keys.
    # KuzuDB might return all properties flat, and labels are implicit by table.
    # This helper might need to become provider-aware or be simplified if providers return Pydantic models directly.
    # For now, assuming a Neo4j-like record structure.
    
    # Defaulting labels if not present, as some providers (like Kuzu) define labels by table.
    labels = record.get('labels', ['Entity']) 
    attributes = record.get('attributes', {})

    entity_node = EntityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record.get('group_id'), # Use .get for safety
        labels=labels,
        created_at=record['created_at'].to_native() if hasattr(record['created_at'], 'to_native') else record['created_at'],
        summary=record.get('summary'),
        attributes=attributes,
        name_embedding=record.get('name_embedding') # Add name_embedding
    )

    # Clean up attributes that are already top-level fields in EntityNode
    # This logic was specific to Neo4j records where all properties were dumped into 'attributes'.
    # If providers return cleaner data, this might not be needed or needs adjustment.
    if isinstance(entity_node.attributes, dict):
        for key_to_pop in ['uuid', 'name', 'group_id', 'name_embedding', 'summary', 'created_at', 'labels']:
            entity_node.attributes.pop(key_to_pop, None)

    return entity_node


def get_community_node_from_record(record: Any) -> CommunityNode:
    return CommunityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record.get('group_id'),
        name_embedding=record.get('name_embedding'), # Make sure this is included by provider query
        created_at=record['created_at'].to_native() if hasattr(record['created_at'], 'to_native') else record['created_at'],
        summary=record.get('summary'),
        labels=record.get('labels', ['Community']) # Default labels
    )


async def create_entity_node_embeddings(embedder: EmbedderClient, nodes: list[EntityNode]):
    # This utility function might be better placed elsewhere or used by providers internally if needed.
    # Or called by Graphiti class before passing nodes to provider's bulk save.
    name_embeddings_list = await embedder.create_batch([node.name for node in nodes])
    for node, name_embedding_list_item in zip(nodes, name_embeddings_list, strict=True):
        if name_embedding_list_item and isinstance(name_embedding_list_item, list):
             node.name_embedding = name_embedding_list_item # Assuming create_batch returns List[List[float]]
        else: # Fallback if structure is not List[List[float]] but List[float] for a single item batch
             node.name_embedding = name_embedding_list_item # This assignment might be incorrect depending on actual return type
        created_at=record['created_at'].to_native().timestamp(),
        valid_at=(record['valid_at'].to_native()),
        uuid=record['uuid'],
        group_id=record['group_id'],
        source=EpisodeType.from_str(record['source']),
        name=record['name'],
        source_description=record['source_description'],
        entity_edges=record['entity_edges'],
    )


def get_entity_node_from_record(record: Any) -> EntityNode:
    entity_node = EntityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        labels=record['labels'],
        created_at=record['created_at'].to_native(),
        summary=record['summary'],
        attributes=record['attributes'],
    )

    entity_node.attributes.pop('uuid', None)
    entity_node.attributes.pop('name', None)
    entity_node.attributes.pop('group_id', None)
    entity_node.attributes.pop('name_embedding', None)
    entity_node.attributes.pop('summary', None)
    entity_node.attributes.pop('created_at', None)

    return entity_node


def get_community_node_from_record(record: Any) -> CommunityNode:
    return CommunityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        name_embedding=record['name_embedding'],
        created_at=record['created_at'].to_native(),
        summary=record['summary'],
    )


async def create_entity_node_embeddings(embedder: EmbedderClient, nodes: list[EntityNode]):
    name_embeddings = await embedder.create_batch([node.name for node in nodes])
    for node, name_embedding in zip(nodes, name_embeddings, strict=True):
        node.name_embedding = name_embedding
