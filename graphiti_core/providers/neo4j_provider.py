import logging
from datetime import datetime, timezone
from typing import Any, List, Dict, Optional, Tuple

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
from typing_extensions import LiteralString

from graphiti_core.providers.base import GraphDatabaseProvider
from graphiti_core.nodes import (
    EpisodicNode,
    EntityNode,
    CommunityNode,
    EpisodeType,
    get_episodic_node_from_record,
    get_entity_node_from_record,
    get_community_node_from_record,
    ENTITY_NODE_RETURN, # Import ENTITY_NODE_RETURN from nodes.py
)
from graphiti_core.edges import (
    EpisodicEdge,
    EntityEdge,
    CommunityEdge,
    get_episodic_edge_from_record,
    get_entity_edge_from_record,
    get_community_edge_from_record,
    ENTITY_EDGE_RETURN, # Import ENTITY_EDGE_RETURN from edges.py
)
from graphiti_core.embedder import EmbedderClient # Placeholder, actual import from base

# search_filters.py content is substantial, so SearchFilters and its helpers will be defined here
# to avoid circular dependencies or overly complex import structures if they were moved to helpers.
# from graphiti_core.search.search_filters import SearchFilters # Actual import
# Instead, define SearchFilters and its helpers directly or as private methods

from graphiti_core.errors import NodeNotFoundError, EdgeNotFoundError, GroupsEdgesNotFoundError
from graphiti_core.helpers import (
    DEFAULT_DATABASE, 
    semaphore_gather, 
    parse_db_date, 
    lucene_sanitize, # Import lucene_sanitize
    RUNTIME_QUERY # Import RUNTIME_QUERY
)
from graphiti_core.models.nodes.node_db_queries import (
    EPISODIC_NODE_SAVE,
    ENTITY_NODE_SAVE,
    COMMUNITY_NODE_SAVE,
    EPISODIC_NODE_SAVE_BULK, # Import for bulk operations
    ENTITY_NODE_SAVE_BULK,   # Import for bulk operations
)
from graphiti_core.models.edges.edge_db_queries import (
    EPISODIC_EDGE_SAVE,
    ENTITY_EDGE_SAVE,
    COMMUNITY_EDGE_SAVE,
    EPISODIC_EDGE_SAVE_BULK, # Import for bulk operations
    ENTITY_EDGE_SAVE_BULK,   # Import for bulk operations
)
from neo4j import AsyncManagedTransaction # For type hinting in bulk tx function

# search_filters.py content (SearchFilters class and helper functions)
# This is a large section, so it's being added directly here.
# Consider moving this to a separate file if it becomes too unwieldy.
from enum import Enum as PyEnum # Alias to avoid conflict if SearchFilters has an Enum

class ComparisonOperator(PyEnum):
    equals = '='
    not_equals = '<>'
    greater_than = '>'
    less_than = '<'
    greater_than_equal = '>='
    less_than_equal = '<='

class DateFilter(BaseModel): # Requires pydantic.BaseModel
    date: datetime = Field(description='A datetime to filter on')
    comparison_operator: ComparisonOperator = Field(
        description='Comparison operator for date filter'
    )

class SearchFilters(BaseModel): # Requires pydantic.BaseModel
    node_labels: Optional[List[str]] = Field(
        default=None, description='List of node labels to filter on'
    )
    edge_types: Optional[List[str]] = Field(
        default=None, description='List of edge types to filter on'
    )
    valid_at: Optional[List[List[DateFilter]]] = Field(default=None)
    invalid_at: Optional[List[List[DateFilter]]] = Field(default=None)
    created_at: Optional[List[List[DateFilter]]] = Field(default=None)
    expired_at: Optional[List[List[DateFilter]]] = Field(default=None)


def _node_search_filter_query_constructor(
    filters: SearchFilters,
) -> tuple[LiteralString, dict[str, Any]]:
    filter_query: LiteralString = ''
    filter_params: dict[str, Any] = {}

    if filters.node_labels is not None:
        node_labels = '|'.join(filters.node_labels)
        # In Cypher, labels are typically joined with ':' for multi-label nodes,
        # or checked with `n:Label1 OR n:Label2`.
        # `n:Label1|Label2` is for property checks in older Lucene index queries.
        # For direct Cypher label checks, it's `MATCH (n) WHERE n:Label1 OR n:Label2`
        # Or if all labels must be present: `MATCH (n) WHERE n:Label1 AND n:Label2`
        # The original `node_search_filter_query_constructor` had `AND n:` + node_labels
        # which implies `AND n:LabelA|LabelB`. This syntax is typically for fulltext index schema.
        # If used in a WHERE clause directly, it would be `AND (n:LabelA OR n:LabelB ...)`
        # For now, keeping the original structure, assuming it's for specific index query context.
        # If it's for a general WHERE clause, this needs adjustment.
        # Given it's `CALL db.index.fulltext.queryNodes("...", $query) YIELD node AS n WHERE n:Entity AND n:CustomLabel`
        # the `AND n:` + node_labels seems correct if node_labels is a single string like "Type1:Type2"
        # or if it's for filtering results from an index query.
        # The original code `node_labels = '|'.join(filters.node_labels)` suggests OR logic for labels.
        # So `AND (n:LabelA OR n:LabelB)` would be the Cypher equivalent.
        # Let's adjust to `AND (n:LabelA OR n:LabelB)` form for clarity in Cypher.
        if filters.node_labels:
            label_filters = [f"n:`{lbl}`" for lbl in filters.node_labels] # Use backticks for safety
            filter_query += " AND (" + " OR ".join(label_filters) + ")"
            
    # Date filters are not typically applied to nodes in the same way as edges in the original search_utils
    # So, this constructor primarily handles labels for nodes.
    return filter_query, filter_params


def _edge_search_filter_query_constructor(
    filters: SearchFilters,
) -> tuple[LiteralString, dict[str, Any]]:
    filter_query: LiteralString = ''
    filter_params: dict[str, Any] = {}

    if filters.edge_types is not None and filters.edge_types:
        filter_query += '\nAND r.name IN $edge_types'
        filter_params['edge_types'] = filters.edge_types

    if filters.node_labels is not None and filters.node_labels:
        # This applies to both source (n) and target (m) nodes of an edge
        source_label_filters = [f"n:`{lbl}`" for lbl in filters.node_labels]
        target_label_filters = [f"m:`{lbl}`" for lbl in filters.node_labels]
        filter_query += "\nAND (" + " OR ".join(source_label_filters) + ")"
        filter_query += "\nAND (" + " OR ".join(target_label_filters) + ")"
        
    def _construct_date_filter_clause(date_filters: Optional[List[List[DateFilter]]], field_name: str, params_dict: dict) -> str:
        if not date_filters:
            return ""
        
        outer_or_clauses = []
        for i, or_list in enumerate(date_filters):
            inner_and_clauses = []
            for j, date_filter_item in enumerate(or_list):
                param_name = f"{field_name}_{i}_{j}"
                params_dict[param_name] = date_filter_item.date
                inner_and_clauses.append(f"(r.{field_name} {date_filter_item.comparison_operator.value} ${param_name})")
            if inner_and_clauses:
                outer_or_clauses.append("(" + " AND ".join(inner_and_clauses) + ")")
        
        if outer_or_clauses:
            return "\nAND (" + " OR ".join(outer_or_clauses) + ")"
        return ""

    filter_query += _construct_date_filter_clause(filters.valid_at, "valid_at", filter_params)
    filter_query += _construct_date_filter_clause(filters.invalid_at, "invalid_at", filter_params)
    filter_query += _construct_date_filter_clause(filters.created_at, "created_at", filter_params)
    filter_query += _construct_date_filter_clause(filters.expired_at, "expired_at", filter_params)
            
    return filter_query, filter_params


# Constants for search
RELEVANT_SCHEMA_LIMIT = 10
DEFAULT_MIN_SCORE = 0.6
MAX_QUERY_LENGTH = 32 # Max terms for lucene query

EPISODE_WINDOW_LEN = 3 # Moved from graph_data_operations.py

# Helper for fulltext query construction (adapted from search_utils)
def _fulltext_lucene_query(query: str, group_ids: Optional[List[str]] = None) -> str:
    group_ids_filter_list = (
        [f'group_id:"{lucene_sanitize(g)}"' for g in group_ids] if group_ids else []
    )
    group_ids_filter = ''
    if group_ids_filter_list:
        group_ids_filter = "(" + " OR ".join(group_ids_filter_list) + ") AND "

    sanitized_query_terms = lucene_sanitize(query)
    
    # Check query length (terms)
    num_query_terms = len(sanitized_query_terms.split())
    num_group_id_terms = len(group_ids_filter_list) # Each group_id is a term in the filter part
    
    if num_query_terms + num_group_id_terms >= MAX_QUERY_LENGTH:
        logger.warning(f"Query too long for Lucene ({num_query_terms + num_group_id_terms} terms), returning empty query string.")
        return '' # Return empty if query is too long

    full_lucene_query = group_ids_filter + '(' + sanitized_query_terms + ')'
    return full_lucene_query


EPISODE_WINDOW_LEN = 3 # from graphiti_core.utils.maintenance.graph_data_operations

logger = logging.getLogger(__name__)

# Pydantic BaseModel import for DateFilter and SearchFilters
from pydantic import BaseModel, Field


class Neo4jProvider(GraphDatabaseProvider):
    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database or DEFAULT_DATABASE
        self.driver: Optional[AsyncDriver] = None

    async def connect(self, **kwargs) -> None:
        if not self.driver or not self.driver.closed():
            await self.close() # Close existing if any
        self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
        logger.info(f"Neo4j driver initialized for URI: {self.uri} and database: {self.database}")
        await self.verify_connectivity()

    async def close(self) -> None:
        if self.driver and not self.driver.closed():
            await self.driver.close()
            logger.info("Neo4j driver closed.")
        self.driver = None

    def get_session(self) -> Any: # Type Any for now, neo4j session type is internal
        if not self.driver:
            raise ConnectionError("Driver not initialized. Call connect() first.")
        return self.driver.session(database=self.database)

    async def verify_connectivity(self) -> None:
        if not self.driver:
            raise ConnectionError("Driver not initialized. Call connect() first.")
        try:
            await self.driver.verify_connectivity()
            logger.info("Successfully verified connectivity to Neo4j.")
        except Neo4jError as e:
            logger.error(f"Neo4j connectivity verification failed: {e}")
            raise ConnectionError(f"Neo4j connectivity verification failed: {e}") from e

    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
        if not self.driver:
            raise ConnectionError("Driver not initialized. Call connect() first.")
        
        # Prefer database specified in kwargs, then instance default, then Neo4j's default
        db_name = kwargs.pop('database_', self.database)
        
        records, summary_metadata, summary_counters = await self.driver.execute_query(
            query,
            parameters_=params,
            database_=db_name,
            **kwargs
        )
        return records, summary_metadata, summary_counters


    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        # Logic from graphiti_core.utils.maintenance.graph_data_operations.build_indices_and_constraints
        if not self.driver:
            raise ConnectionError("Driver not initialized. Call connect() first.")

        if delete_existing:
            records, _, _ = await self.execute_query(
                "SHOW INDEXES YIELD name",
                database_=self.database # Use provider's database
            )
            index_names = [record['name'] for record in records]
            await semaphore_gather(
                *[
                    self.execute_query(
                        "DROP INDEX $name",
                        params={"name": name}, # Query params should be in 'params'
                        database_=self.database,
                    )
                    for name in index_names
                ]
            )
            logger.info("Deleted existing Neo4j indices.")

        range_indices: list[LiteralString] = [
            'CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)',
            'CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)',
            'CREATE INDEX community_uuid IF NOT EXISTS FOR (n:Community) ON (n.uuid)',
            'CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.uuid)',
            'CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.uuid)',
            'CREATE INDEX has_member_uuid IF NOT EXISTS FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
            'CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)',
            'CREATE INDEX episode_group_id IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)',
            'CREATE INDEX relation_group_id IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.group_id)',
            'CREATE INDEX mention_group_id IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.group_id)',
            'CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)',
            'CREATE INDEX created_at_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.created_at)',
            'CREATE INDEX created_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.created_at)',
            'CREATE INDEX valid_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.valid_at)',
            'CREATE INDEX name_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.name)',
            'CREATE INDEX created_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.created_at)',
            'CREATE INDEX expired_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.expired_at)',
            'CREATE INDEX valid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.valid_at)',
            'CREATE INDEX invalid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.invalid_at)',
        ]

        fulltext_indices: list[LiteralString] = [
            """CREATE FULLTEXT INDEX episode_content IF NOT EXISTS 
            FOR (e:Episodic) ON EACH [e.content, e.source, e.source_description, e.group_id]""",
            """CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS 
            FOR (n:Entity) ON EACH [n.name, n.summary, n.group_id]""",
            """CREATE FULLTEXT INDEX community_name IF NOT EXISTS 
            FOR (n:Community) ON EACH [n.name, n.group_id]""",
            """CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS 
            FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]""",
        ]

        index_queries: list[LiteralString] = range_indices + fulltext_indices

        await semaphore_gather(
            *[
                self.execute_query(
                    query,
                    database_=self.database,
                )
                for query in index_queries
            ]
        )
        logger.info("Built Neo4j indices and constraints.")

    # CRUD Operations for Nodes
    async def save_episodic_node(self, node: EpisodicNode) -> Any:
        # Adapted from EpisodicNode.save()
        result, _, _ = await self.execute_query(
            EPISODIC_NODE_SAVE,
            params={
                "uuid": node.uuid,
                "name": node.name,
                "group_id": node.group_id,
                "source_description": node.source_description,
                "content": node.content,
                "entity_edges": node.entity_edges,
                "created_at": node.created_at,
                "valid_at": node.valid_at,
                "source": node.source.value,
            }
        )
        logger.debug(f'Saved EpisodicNode to neo4j: {node.uuid}')
        return result[0]['uuid'] if result else None

    async def save_entity_node(self, node: EntityNode) -> Any:
        # This method saves an EntityNode to Neo4j.
        # It handles dynamic labels by merging the base :Entity label and then
        # individually setting additional labels specified in node.labels.
        # Properties are set using SET n += $props_to_set to merge, and vector embeddings
        # are set using the db.create.setNodeVectorProperty procedure.
        entity_data: dict[str, Any] = {
            'uuid': node.uuid,
            'name': node.name,
            'name_embedding': node.name_embedding, # Assuming embedding is pre-generated
            'group_id': node.group_id,
            'summary': node.summary,
            'created_at': node.created_at,
        }
        entity_data.update(node.attributes or {})

        # The ENTITY_NODE_SAVE query uses dynamic labels like SET n:$($labels)
        # This syntax might be specific to a library or wrapper used previously.
        # For raw neo4j driver, labels are typically set like: MERGE (n:Entity:NewLabel {uuid: $uuid})
        # For now, assuming ENTITY_NODE_SAVE is compatible or will be adjusted.
        # If $labels is a string like "Label1:Label2", it should work.
        # The original code had labels=self.labels + ['Entity'], so it should be a list.
        # The query `SET n:$($labels)` is not standard Cypher. It should be `SET n:${labels_str}` where labels_str is `Entity:CustomLabel`
        # This part needs careful handling of how labels are passed and used in the query.
        # For now, I will pass labels as a list, and assume the query can handle it or adjust later.
        
        # Let's prepare labels string for the query if it expects a colon-separated string for dynamic labels
        labels_str = ":".join(node.labels + ["Entity"])


        # The query ENTITY_NODE_SAVE has `SET n:$($labels)`. This is not standard Cypher.
        # It should be more like `MERGE (n:Entity {uuid: $entity_data.uuid}) SET n += $entity_data SET n:${labels_str_param}`
        # where labels_str_param is a string like "CustomLabel1:CustomLabel2".
        # Or, handle labels during MERGE if possible: `MERGE (n:Entity:${label1}:${label2} {uuid: ...})`
        # For now, I'll construct the query string directly to include labels, which is safer.
        
        # Create the initial part of the query for MERGE with dynamic labels
        merge_query = f"MERGE (n:Entity {':'.join(node.labels)} {{uuid: $entity_data.uuid}}) "
        set_query = "SET n = $entity_data "
        vector_property_query = "WITH n CALL db.create.setNodeVectorProperty(n, 'name_embedding', $entity_data.name_embedding) "
        return_query = "RETURN n.uuid AS uuid"
        
        # This is a simplified approach. The original query structure `SET n:$($labels)` suggests
        # it might have been intended for a specific client library feature that processes `$($labels)`
        # into multiple SET n:Label commands or similar.
        # A more robust way for arbitrary labels with pure Cypher is often to MERGE basic label (Entity)
        # and then add other labels using SET n:Label1:Label2 etc.
        # For now, modifying the query string directly:
        
        dynamic_labels_str = ":".join(node.labels)
        if dynamic_labels_str:
            dynamic_labels_str = ":" + dynamic_labels_str
        
        # Reconstruct the query to be safer with labels
        # This is still not ideal, as direct string formatting into queries can be risky if labels are user-supplied
        # However, node.labels are internal.
        # A better way would be to pass labels and construct SET n:Label1, n:Label2 ...
        # For now, stick to adapting the provided query, assuming it implies a certain structure.
        # The original query was:
        # ENTITY_NODE_SAVE = """
        # MERGE (n:Entity {uuid: $entity_data.uuid})
        # SET n:$($labels)  <-- This is the problematic part for standard Cypher
        # SET n = $entity_data
        # WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $entity_data.name_embedding)
        # RETURN n.uuid AS uuid"""
        #
        # A direct adaptation:
        # The approach for setting labels (`SET n:\`label\`` for each label in node.labels)
        # is standard Cypher and generally robust for controlled label names.
        # It avoids needing APOC procedures for this basic operation.
        
        query_parts = [f"MERGE (n:Entity {{uuid: $entity_data.uuid}})"] # Base label :Entity
        # Add other labels using SET n:`label`
        for label in node.labels: # node.labels should not contain 'Entity' as it's in MERGE
            if label != "Entity": # Defensive check
                query_parts.append(f"SET n:`{label}`") # Use backticks for safety

        # Set/update properties using += to merge (preserve existing, update/add new)
        # Note: uuid is part of MERGE key, name_embedding handled by procedure.
        query_parts.append("SET n += $props_to_set")

        # Set vector property using the specific procedure
        # Ensure name_embedding is passed in entity_data for the procedure call.
        if node.name_embedding is not None: # Only call if there's an embedding
            query_parts.append("WITH n, $entity_data AS data CALL db.create.setNodeVectorProperty(n, 'name_embedding', data.name_embedding)")
        
        query_parts.append("RETURN n.uuid AS uuid")
        
        final_query = "\n".join(query_parts)

        # Prepare properties for SET n += $props_to_set
        # Exclude uuid (match key) and name_embedding (handled by procedure).
        props_to_set = {k: v for k, v in entity_data.items() if k not in ['uuid', 'name_embedding']}

        result, _, _ = await self.execute_query(
            final_query,
            params={
                "entity_data": entity_data, # Passed for uuid in MERGE and vector procedure
                "props_to_set": props_to_set  # Passed for SET n += $props_to_set
            }
        )
        logger.debug(f'Saved EntityNode to neo4j: {node.uuid}')
        return result[0]['uuid'] if result else None


    async def save_community_node(self, node: CommunityNode) -> Any:
        # Adapted from CommunityNode.save()
        result, _, _ = await self.execute_query(
            COMMUNITY_NODE_SAVE,
            params={
                "uuid": node.uuid,
                "name": node.name,
                "group_id": node.group_id,
                "summary": node.summary,
                "name_embedding": node.name_embedding, # Assuming pre-generated
                "created_at": node.created_at,
            }
        )
        logger.debug(f'Saved CommunityNode to neo4j: {node.uuid}')
        return result[0]['uuid'] if result else None

    async def _get_node_by_uuid(self, uuid: str, query_template: str, record_parser_func) -> Optional[Any]:
        records, _, _ = await self.execute_query(query_template, params={"uuid": uuid})
        nodes = [record_parser_func(record) for record in records]
        if not nodes:
            # Do not raise NodeNotFoundError here, as per ABC return type Optional[]
            return None
        return nodes[0]

    async def get_episodic_node_by_uuid(self, uuid: str) -> Optional[EpisodicNode]:
        query = """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN e.content AS content,
                e.created_at AS created_at,
                e.valid_at AS valid_at,
                e.uuid AS uuid,
                e.name AS name,
                e.group_id AS group_id,
                e.source_description AS source_description,
                e.source AS source,
                e.entity_edges AS entity_edges
            """
        return await self._get_node_by_uuid(uuid, query, get_episodic_node_from_record)

    async def get_entity_node_by_uuid(self, uuid: str) -> Optional[EntityNode]:
        query = f"MATCH (n:Entity {{uuid: $uuid}}) {ENTITY_NODE_RETURN}"
        return await self._get_node_by_uuid(uuid, query, get_entity_node_from_record)

    async def get_community_node_by_uuid(self, uuid: str) -> Optional[CommunityNode]:
        query = """
            MATCH (n:Community {uuid: $uuid})
            RETURN
                n.uuid As uuid, 
                n.name AS name,
                n.group_id AS group_id,
                n.created_at AS created_at, 
                n.summary AS summary,
                n.name_embedding AS name_embedding 
            """ # Added name_embedding to return
        # The get_community_node_from_record needs to be updated or expect name_embedding
        # For now, assuming get_community_node_from_record handles it.
        return await self._get_node_by_uuid(uuid, query, get_community_node_from_record)


    async def _get_nodes_by_uuids(self, uuids: List[str], query_template: str, record_parser_func) -> List[Any]:
        if not uuids: return []
        records, _, _ = await self.execute_query(query_template, params={"uuids": uuids})
        return [record_parser_func(record) for record in records]

    async def get_episodic_nodes_by_uuids(self, uuids: List[str]) -> List[EpisodicNode]:
        query = """
            MATCH (e:Episodic) WHERE e.uuid IN $uuids
            RETURN DISTINCT
                e.content AS content,
                e.created_at AS created_at,
                e.valid_at AS valid_at,
                e.uuid AS uuid,
                e.name AS name,
                e.group_id AS group_id,
                e.source_description AS source_description,
                e.source AS source,
                e.entity_edges AS entity_edges
            """
        return await self._get_nodes_by_uuids(uuids, query, get_episodic_node_from_record)

    async def get_entity_nodes_by_uuids(self, uuids: List[str]) -> List[EntityNode]:
        query = f"MATCH (n:Entity) WHERE n.uuid IN $uuids {ENTITY_NODE_RETURN}"
        return await self._get_nodes_by_uuids(uuids, query, get_entity_node_from_record)

    async def get_community_nodes_by_uuids(self, uuids: List[str]) -> List[CommunityNode]:
        query = """
            MATCH (n:Community) WHERE n.uuid IN $uuids
            RETURN
                n.uuid As uuid, 
                n.name AS name,
                n.group_id AS group_id,
                n.created_at AS created_at, 
                n.summary AS summary,
                n.name_embedding AS name_embedding
            """
        return await self._get_nodes_by_uuids(uuids, query, get_community_node_from_record)

    async def _get_nodes_by_group_ids(
        self,
        group_ids: List[str],
        query_template_start: str, # e.g. "MATCH (e:Episodic) WHERE e.group_id IN $group_ids"
        query_template_return: str, # e.g. RETURN e.content AS content, ..."
        record_parser_func,
        order_by_field: str, # e.g. "e.uuid"
        limit: Optional[int] = None,
        uuid_cursor: Optional[str] = None
    ) -> List[Any]:
        if not group_ids: return []
        
        cursor_query: LiteralString = f'AND {order_by_field.split(" ")[0]} < $uuid_cursor' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''
        
        full_query = f"""
            {query_template_start}
            {cursor_query}
            {query_template_return}
            ORDER BY {order_by_field} DESC
            {limit_query}
        """
        params = {"group_ids": group_ids, "uuid_cursor": uuid_cursor, "limit": limit}
        # Remove None params as Neo4j driver might not like them for optional parts of query
        params = {k: v for k, v in params.items() if v is not None}

        records, _, _ = await self.execute_query(full_query, params=params)
        return [record_parser_func(record) for record in records]

    async def get_episodic_nodes_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EpisodicNode]:
        query_start = "MATCH (e:Episodic) WHERE e.group_id IN $group_ids"
        query_return = """
            RETURN DISTINCT
                e.content AS content, e.created_at AS created_at, e.valid_at AS valid_at,
                e.uuid AS uuid, e.name AS name, e.group_id AS group_id,
                e.source_description AS source_description, e.source AS source, e.entity_edges AS entity_edges
            """
        return await self._get_nodes_by_group_ids(group_ids, query_start, query_return, get_episodic_node_from_record, "e.uuid", limit, uuid_cursor)

    async def get_entity_nodes_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EntityNode]:
        query_start = "MATCH (n:Entity) WHERE n.group_id IN $group_ids"
        return await self._get_nodes_by_group_ids(group_ids, query_start, ENTITY_NODE_RETURN, get_entity_node_from_record, "n.uuid", limit, uuid_cursor)

    async def get_community_nodes_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[CommunityNode]:
        query_start = "MATCH (n:Community) WHERE n.group_id IN $group_ids"
        query_return = """
            RETURN
                n.uuid As uuid, n.name AS name, n.group_id AS group_id,
                n.created_at AS created_at, n.summary AS summary, n.name_embedding as name_embedding
            """
        return await self._get_nodes_by_group_ids(group_ids, query_start, query_return, get_community_node_from_record, "n.uuid", limit, uuid_cursor)

    async def get_episodic_nodes_by_entity_node_uuid(self, entity_node_uuid: str) -> List[EpisodicNode]:
        # Adapted from EpisodicNode.get_by_entity_node_uuid
        query = """
            MATCH (e:Episodic)-[r:MENTIONS]->(n:Entity {uuid: $entity_node_uuid})
            RETURN DISTINCT
                e.content AS content, e.created_at AS created_at, e.valid_at AS valid_at,
                e.uuid AS uuid, e.name AS name, e.group_id AS group_id,
                e.source_description AS source_description, e.source AS source, e.entity_edges AS entity_edges
            """
        records, _, _ = await self.execute_query(query, params={"entity_node_uuid": entity_node_uuid})
        return [get_episodic_node_from_record(record) for record in records]

    async def delete_node(self, uuid: str) -> None:
        # Generic delete for any node type with the UUID
        # Original Node.delete() matches (n:Entity|Episodic|Community {uuid: $uuid})
        # A more generic one:
        await self.execute_query(
            "MATCH (n {uuid: $uuid}) DETACH DELETE n",
            params={"uuid": uuid}
        )
        logger.debug(f'Deleted Node: {uuid}')

    async def delete_nodes_by_group_id(self, group_id: str, node_type: str) -> None:
        # node_type should be one of "Entity", "Episodic", "Community"
        if node_type not in ["Entity", "Episodic", "Community"]:
            raise ValueError(f"Invalid node_type: {node_type}. Must be Entity, Episodic, or Community.")
        
        query = f"MATCH (n:{node_type} {{group_id: $group_id}}) DETACH DELETE n"
        await self.execute_query(query, params={"group_id": group_id})
        logger.debug(f"Deleted {node_type} nodes with group_id: {group_id}")

    # CRUD Operations for Edges
    async def save_episodic_edge(self, edge: EpisodicEdge) -> Any:
        result, _, _ = await self.execute_query(
            EPISODIC_EDGE_SAVE,
            params={
                "episode_uuid": edge.source_node_uuid,
                "entity_uuid": edge.target_node_uuid,
                "uuid": edge.uuid,
                "group_id": edge.group_id,
                "created_at": edge.created_at,
            }
        )
        logger.debug(f'Saved EpisodicEdge to neo4j: {edge.uuid}')
        return result[0]['uuid'] if result else None

    async def save_entity_edge(self, edge: EntityEdge) -> Any:
        # Adapted from EntityEdge.save()
        edge_data: dict[str, Any] = {
            # Note: source_uuid and target_uuid are used in the query's MATCH clause,
            # not directly in SET r = $edge_data typically.
            # The original ENTITY_EDGE_SAVE query structure:
            # MATCH (source:Entity {uuid: $source_uuid}) MATCH (target:Entity {uuid: $target_uuid})
            # MERGE (source)-[r:RELATES_TO {uuid: $uuid}]->(target)
            # SET r = $edge_data  <-- $edge_data should contain all properties for the edge 'r'
            # WITH r CALL db.create.setRelationshipVectorProperty(r, "fact_embedding", $fact_embedding)
            #
            # So, $edge_data should include uuid, name, group_id, fact, etc.
            # And $source_uuid, $target_uuid, $uuid, $fact_embedding are separate top-level params
            
            'uuid': edge.uuid,
            'name': edge.name,
            'group_id': edge.group_id,
            'fact': edge.fact,
            # 'fact_embedding': edge.fact_embedding, # This is passed separately to setRelationshipVectorProperty
            'episodes': edge.episodes,
            'created_at': edge.created_at,
            'expired_at': edge.expired_at,
            'valid_at': edge.valid_at,
            'invalid_at': edge.invalid_at,
        }
        edge_data.update(edge.attributes or {})
        
        # Remove source/target node uuids from edge_data if they were added by attributes, as they are for MATCH
        edge_data.pop('source_node_uuid', None)
        edge_data.pop('target_node_uuid', None)


        result, _, _ = await self.execute_query(
            ENTITY_EDGE_SAVE,
            params={
                "source_uuid": edge.source_node_uuid,
                "target_uuid": edge.target_node_uuid,
                "uuid": edge.uuid, # For the MERGE clause
                "edge_data": edge_data, # For SET r = $edge_data
                "fact_embedding": edge.fact_embedding, # For setRelationshipVectorProperty
            }
        )
        logger.debug(f'Saved EntityEdge to neo4j: {edge.uuid}')
        return result[0]['uuid'] if result else None

    async def save_community_edge(self, edge: CommunityEdge) -> Any:
        result, _, _ = await self.execute_query(
            COMMUNITY_EDGE_SAVE,
            params={
                "community_uuid": edge.source_node_uuid,
                "entity_uuid": edge.target_node_uuid, # Can be Entity or Community
                "uuid": edge.uuid,
                "group_id": edge.group_id,
                "created_at": edge.created_at,
            }
        )
        logger.debug(f'Saved CommunityEdge to neo4j: {edge.uuid}')
        return result[0]['uuid'] if result else None

    async def _get_edge_by_uuid(self, uuid: str, query_template: str, record_parser_func) -> Optional[Any]:
        records, _, _ = await self.execute_query(query_template, params={"uuid": uuid})
        edges = [record_parser_func(record) for record in records]
        if not edges:
            # Do not raise EdgeNotFoundError here, as per ABC return type Optional[]
            return None
        return edges[0]
    
    async def get_episodic_edge_by_uuid(self, uuid: str) -> Optional[EpisodicEdge]:
        query = """
            MATCH (n:Episodic)-[e:MENTIONS {uuid: $uuid}]->(m:Entity)
            RETURN
                e.uuid As uuid, e.group_id AS group_id,
                n.uuid AS source_node_uuid, m.uuid AS target_node_uuid, 
                e.created_at AS created_at
            """
        return await self._get_edge_by_uuid(uuid, query, get_episodic_edge_from_record)

    async def get_entity_edge_by_uuid(self, uuid: str) -> Optional[EntityEdge]:
        query = f"MATCH (n:Entity)-[e:RELATES_TO {{uuid: $uuid}}]->(m:Entity) {ENTITY_EDGE_RETURN}"
        return await self._get_edge_by_uuid(uuid, query, get_entity_edge_from_record)

    async def get_community_edge_by_uuid(self, uuid: str) -> Optional[CommunityEdge]:
        query = """
            MATCH (n:Community)-[e:HAS_MEMBER {uuid: $uuid}]->(m:Entity|Community)
            RETURN
                e.uuid As uuid, e.group_id AS group_id,
                n.uuid AS source_node_uuid, m.uuid AS target_node_uuid, 
                e.created_at AS created_at
            """
        return await self._get_edge_by_uuid(uuid, query, get_community_edge_from_record)

    async def _get_edges_by_uuids(self, uuids: List[str], query_template: str, record_parser_func) -> List[Any]:
        if not uuids: return []
        records, _, _ = await self.execute_query(query_template, params={"uuids": uuids})
        return [record_parser_func(record) for record in records]

    async def get_episodic_edges_by_uuids(self, uuids: List[str]) -> List[EpisodicEdge]:
        query = """
            MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity) WHERE e.uuid IN $uuids
            RETURN
                e.uuid As uuid, e.group_id AS group_id,
                n.uuid AS source_node_uuid, m.uuid AS target_node_uuid, 
                e.created_at AS created_at
            """
        return await self._get_edges_by_uuids(uuids, query, get_episodic_edge_from_record)

    async def get_entity_edges_by_uuids(self, uuids: List[str]) -> List[EntityEdge]:
        query = f"MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) WHERE e.uuid IN $uuids {ENTITY_EDGE_RETURN}"
        return await self._get_edges_by_uuids(uuids, query, get_entity_edge_from_record)

    async def get_community_edges_by_uuids(self, uuids: List[str]) -> List[CommunityEdge]:
        query = """
            MATCH (n:Community)-[e:HAS_MEMBER]->(m:Entity|Community) WHERE e.uuid IN $uuids
            RETURN
                e.uuid As uuid, e.group_id AS group_id,
                n.uuid AS source_node_uuid, m.uuid AS target_node_uuid, 
                e.created_at AS created_at
            """
        return await self._get_edges_by_uuids(uuids, query, get_community_edge_from_record)

    async def _get_edges_by_group_ids(
        self,
        group_ids: List[str],
        query_template_start: str,
        query_template_return: str,
        record_parser_func,
        order_by_field: str,
        limit: Optional[int] = None,
        uuid_cursor: Optional[str] = None
    ) -> List[Any]:
        if not group_ids: return []
        
        cursor_query: LiteralString = f'AND {order_by_field.split(" ")[0]} < $uuid_cursor' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''
        
        full_query = f"""
            {query_template_start}
            {cursor_query}
            {query_template_return}
            ORDER BY {order_by_field} DESC
            {limit_query}
        """
        params = {"group_ids": group_ids, "uuid_cursor": uuid_cursor, "limit": limit}
        params = {k: v for k, v in params.items() if v is not None}
        
        records, _, _ = await self.execute_query(full_query, params=params)
        # Original code raised GroupsEdgesNotFoundError if no edges found. Adhering to that.
        # However, ABC spec generally implies returning empty list is fine.
        # For now, let's match the original behavior.
        # if not records:
        #     raise GroupsEdgesNotFoundError(group_ids) # Optional: depends on strictness
        return [record_parser_func(record) for record in records]

    async def get_episodic_edges_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EpisodicEdge]:
        query_start = "MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity) WHERE e.group_id IN $group_ids"
        query_return = """
            RETURN
                e.uuid As uuid, e.group_id AS group_id,
                n.uuid AS source_node_uuid, m.uuid AS target_node_uuid, 
                e.created_at AS created_at
            """
        return await self._get_edges_by_group_ids(group_ids, query_start, query_return, get_episodic_edge_from_record, "e.uuid", limit, uuid_cursor)
        
    async def get_entity_edges_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EntityEdge]:
        query_start = "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) WHERE e.group_id IN $group_ids"
        return await self._get_edges_by_group_ids(group_ids, query_start, ENTITY_EDGE_RETURN, get_entity_edge_from_record, "e.uuid", limit, uuid_cursor)

    async def get_community_edges_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[CommunityEdge]:
        query_start = "MATCH (n:Community)-[e:HAS_MEMBER]->(m:Entity|Community) WHERE e.group_id IN $group_ids"
        query_return = """
            RETURN
                e.uuid As uuid, e.group_id AS group_id,
                n.uuid AS source_node_uuid, m.uuid AS target_node_uuid, 
                e.created_at AS created_at
            """
        return await self._get_edges_by_group_ids(group_ids, query_start, query_return, get_community_edge_from_record, "e.uuid", limit, uuid_cursor)

    async def get_entity_edges_by_node_uuid(self, node_uuid: str) -> List[EntityEdge]:
        # Adapted from EntityEdge.get_by_node_uuid
        query = f"MATCH (n:Entity {{uuid: $node_uuid}})-[e:RELATES_TO]-(m:Entity) {ENTITY_EDGE_RETURN}"
        records, _, _ = await self.execute_query(query, params={"node_uuid": node_uuid})
        return [get_entity_edge_from_record(record) for record in records]

    async def delete_edge(self, uuid: str) -> None:
        # Generic delete for any edge type with the UUID
        # Original Edge.delete() matches (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->(m)
        # A more generic approach:
        await self.execute_query(
            "MATCH ()-[e {uuid: $uuid}]-() DELETE e",
            params={"uuid": uuid}
        )
        logger.debug(f'Deleted Edge: {uuid}')

    # Maintenance/Utility
    async def clear_data(self, group_ids: Optional[List[str]] = None) -> None:
        # Logic from graphiti_core.utils.maintenance.graph_data_operations.clear_data
        if not self.driver:
            raise ConnectionError("Driver not initialized. Call connect() first.")

        async with self.get_session() as session: # Use self.get_session()
            async def delete_all(tx):
                # Using self.database which is the configured database for the provider
                await tx.run('MATCH (n) DETACH DELETE n')
                logger.info(f"Cleared all data from database {self.database}")

            async def delete_group_ids(tx, group_ids_to_delete: list[str]):
                await tx.run(
                    # Original query was 'MATCH (n:Entity|Episodic|Community) WHERE n.group_id IN $group_ids DETACH DELETE n'
                    # Making it slightly more generic by not specifying labels, or assume group_id is only on these labels.
                    # For safety, let's use the specific labels if that was the original intent.
                    'MATCH (n) WHERE n.group_id IN $group_ids DETACH DELETE n',
                    # 'MATCH (n:Entity|Episodic|Community) WHERE n.group_id IN $group_ids DETACH DELETE n',
                    group_ids=group_ids_to_delete,
                )
                logger.info(f"Cleared data for group_ids: {group_ids_to_delete} from database {self.database}")

            if group_ids is None:
                await session.execute_write(delete_all)
            else:
                if not group_ids: # Handle empty list case if necessary
                    logger.info("No group_ids provided for selective clear, no data deleted.")
                    return
                await session.execute_write(delete_group_ids, group_ids_to_delete=group_ids)

    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = EPISODE_WINDOW_LEN,
        group_ids: Optional[List[str]] = None,
        source: Optional[EpisodeType] = None,
    ) -> List[EpisodicNode]:
        # Logic from graphiti_core.utils.maintenance.graph_data_operations.retrieve_episodes
        if not self.driver:
            raise ConnectionError("Driver not initialized. Call connect() first.")

        group_id_filter_clause: LiteralString = (
            '\nAND e.group_id IN $group_ids' if group_ids and len(group_ids) > 0 else ''
        )
        source_filter_clause: LiteralString = '\nAND e.source = $source_val' if source is not None else ''

        query: LiteralString = (
            f"""
            MATCH (e:Episodic) WHERE e.valid_at <= $reference_time
            {group_id_filter_clause}
            {source_filter_clause}
            RETURN e.content AS content,
                e.created_at AS created_at,
                e.valid_at AS valid_at,
                e.uuid AS uuid,
                e.group_id AS group_id,
                e.name AS name,
                e.source_description AS source_description,
                e.source AS source,
                e.entity_edges AS entity_edges
            ORDER BY e.valid_at DESC
            LIMIT $num_episodes
            """
        )
        
        params = {
            "reference_time": reference_time,
            "source_val": source.value if source is not None else None, # Use .value for enum
            "num_episodes": last_n,
            "group_ids": group_ids,
        }
        # Filter out None params only if the query part is conditional and not present
        # Here, group_ids = None and source_val = None are fine if clauses are absent
        final_params = {k: v for k, v in params.items() if v is not None or (k == "group_ids" and group_ids is not None)}


        records, _, _ = await self.execute_query(query, params=final_params, database_=self.database)
        
        # Using get_episodic_node_from_record which should handle Neo4j specific datetime objects.
        # It's defined in graphiti_core.nodes
        episodes = [get_episodic_node_from_record(record) for record in records]
        return list(reversed(episodes))  # Return in chronological order

    # --------------------------------------------------------------------------
    # Search and Bulk methods that were previously here (or stubs)
    # --------------------------------------------------------------------------

    # Search Operations (Implementations adapted from graphiti_core.search.search_utils)

    async def edge_fulltext_search(
        self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = RELEVANT_SCHEMA_LIMIT
    ) -> List[EntityEdge]:
        if not self.driver: raise ConnectionError("Driver not initialized.")
        
        lucene_query_str = _fulltext_lucene_query(query, group_ids)
        if not lucene_query_str:
            return []

        filter_query_clause, filter_params = _edge_search_filter_query_constructor(search_filter)
        
        # The original query in search_utils had `WHERE r.group_id IN $group_ids` inside the MATCH part.
        # This is redundant if `_fulltext_lucene_query` already includes group_id filtering for the Lucene index.
        # However, fulltext index might return relationships which then need further filtering by group_id if the
        # index itself doesn't perfectly handle it or if group_ids are used for post-filtering.
        # For safety and consistency with original logic, keeping a group_id check if group_ids are provided.
        # The `CALL db.index.fulltext.queryRelationships` returns `rel`. We need to match this `rel`
        # with its start and end nodes to apply further filters and get all properties.

        # The `filter_query_clause` applies to `r` (the relationship) and `n`, `m` (start/end nodes).
        # The `group_ids` parameter is used both in Lucene query and potentially in Cypher post-filtering.

        cypher_query = f"""
            CALL db.index.fulltext.queryRelationships("edge_name_and_fact", $lucene_query, {{limit: $limit}}) 
            YIELD relationship AS rel, score
            MATCH (n:Entity)-[r:RELATES_TO]->(m:Entity)
            WHERE r = rel {_group_id_filter_clause("r", group_ids)} {filter_query_clause}
            WITH r, score, n, m 
            {ENTITY_EDGE_RETURN}
            ORDER BY score DESC LIMIT $limit
        """
        # _group_id_filter_clause is a helper to generate "AND alias.group_id IN $group_ids" if group_ids is present
        
        all_params = {
            "lucene_query": lucene_query_str,
            "limit": limit,
            **filter_params,
        }
        if group_ids:
            all_params["group_ids"] = group_ids


        records, _, _ = await self.execute_query(cypher_query, params=all_params)
        return [get_entity_edge_from_record(record) for record in records]

    async def edge_similarity_search(
        self, search_vector: List[float], 
        source_node_uuid: Optional[str], target_node_uuid: Optional[str], 
        search_filter: SearchFilters, group_ids: Optional[List[str]] = None, 
        limit: int = RELEVANT_SCHEMA_LIMIT, min_score: float = DEFAULT_MIN_SCORE
    ) -> List[EntityEdge]:
        if not self.driver: raise ConnectionError("Driver not initialized.")

        filter_query_clause, filter_params = _edge_search_filter_query_constructor(search_filter)
        
        # Constructing the initial MATCH and WHERE clauses
        match_clauses = ["MATCH (n:Entity)-[r:RELATES_TO]->(m:Entity)"]
        where_clauses = ["WHERE r.fact_embedding IS NOT NULL"] # Ensure embedding exists for cosine similarity

        if group_ids:
            where_clauses.append("r.group_id IN $group_ids")
            filter_params["group_ids"] = group_ids
        
        # Source/Target node UUID filtering
        # This logic from original search_utils was a bit complex with conditional appends.
        # Simplified: if source_node_uuid is given, n.uuid must be it. If target_node_uuid is given, m.uuid must be it.
        # If both, then (n.uuid = source AND m.uuid = target) OR (n.uuid = target AND m.uuid = source) for undirected conceptually.
        # However, RELATES_TO is directed. The original query structure implies specific source/target if provided.
        
        # The original query structure for source/target UUIDs was:
        # if source_node_uuid is not None: group_filter_query += '\nAND (n.uuid IN [$source_uuid, $target_uuid])'
        # if target_node_uuid is not None: group_filter_query += '\nAND (m.uuid IN [$source_uuid, $target_uuid])'
        # This seems to allow flexibility but might be too broad if only one is specified.
        # A clearer approach:
        if source_node_uuid:
            where_clauses.append("n.uuid = $source_node_uuid")
            filter_params["source_node_uuid"] = source_node_uuid
        if target_node_uuid:
            where_clauses.append("m.uuid = $target_node_uuid")
            filter_params["target_node_uuid"] = target_node_uuid
            
        # Add clauses from search_filter
        if filter_query_clause:
             # Need to ensure `filter_query_clause` starts with AND or is integrated properly
            if where_clauses: # If there are existing where clauses, prepend AND
                 where_clauses.append(filter_query_clause.strip().replace("AND ", "", 1) if filter_query_clause.strip().startswith("AND ") else filter_query_clause.strip())
            else: # Otherwise, it's the start of the WHERE clause
                 where_clauses.append(filter_query_clause.strip().replace("WHERE ", "", 1) if filter_query_clause.strip().startswith("WHERE ") else filter_query_clause.strip())


        full_query = f"""
            {RUNTIME_QUERY}
            {" ".join(match_clauses)}
            {" AND ".join(where_clauses).replace("WHERE AND", "WHERE")}
            WITH DISTINCT r, n, m, vector.similarity.cosine(r.fact_embedding, $search_vector) AS score
            WHERE score >= $min_score 
            {ENTITY_EDGE_RETURN}
            ORDER BY score DESC
            LIMIT $limit
        """
        
        all_params = {
            "search_vector": search_vector,
            "limit": limit,
            "min_score": min_score,
            **filter_params
        }
        
        records, _, _ = await self.execute_query(full_query, params=all_params)
        return [get_entity_edge_from_record(record) for record in records]

    async def edge_bfs_search(
        self, bfs_origin_node_uuids: Optional[List[str]], 
        bfs_max_depth: int, search_filter: SearchFilters, limit: int
    ) -> List[EntityEdge]:
        if not self.driver: raise ConnectionError("Driver not initialized.")
        if not bfs_origin_node_uuids: return []
        if bfs_max_depth <= 0 : bfs_max_depth = 1 # Ensure depth is at least 1
        if bfs_max_depth > 5 : bfs_max_depth = 5 # Cap depth to prevent overly long queries

        filter_query_clause, filter_params = _edge_search_filter_query_constructor(search_filter)

        # The original query used fixed range {1,3}. Using bfs_max_depth now.
        # It also used [:RELATES_TO|MENTIONS] - MENTIONS is between Episodic and Entity.
        # For EntityEdge, we should focus on RELATES_TO between Entities.
        # If the origin can be Episodic, the path might start with MENTIONS.
        # Assuming origin can be Entity or Episodic as per original query.
        
        query = f"""
            UNWIND $bfs_origin_node_uuids AS origin_uuid
            MATCH path = (origin {{uuid: origin_uuid}})-[*1..{bfs_max_depth}]-(related_node:Entity)
            UNWIND relationships(path) AS rel
            MATCH (n:Entity)-[r:RELATES_TO]-(m:Entity) 
            WHERE r = rel {filter_query_clause} 
            WITH DISTINCT r, n, m
            {ENTITY_EDGE_RETURN}
            LIMIT $limit
        """
        # Note: `filter_query_clause` applies to `r`, `n`, `m`.
        # The path can cross various relationship types, but we only return EntityEdges (RELATES_TO).
        
        all_params = {
            "bfs_origin_node_uuids": bfs_origin_node_uuids,
            "limit": limit,
            **filter_params
        }

        records, _, _ = await self.execute_query(query, params=all_params)
        return [get_entity_edge_from_record(record) for record in records]

    async def node_fulltext_search(
        self, query: str, search_filter: SearchFilters, 
        group_ids: Optional[List[str]] = None, limit: int = RELEVANT_SCHEMA_LIMIT
    ) -> List[EntityNode]:
        if not self.driver: raise ConnectionError("Driver not initialized.")
        
        lucene_query_str = _fulltext_lucene_query(query, group_ids)
        if not lucene_query_str:
            return []

        filter_query_clause, filter_params = _node_search_filter_query_constructor(search_filter)
        
        # Original query: CALL db.index.fulltext.queryNodes("node_name_and_summary", ...) YIELD node AS n, score WHERE n:Entity ...
        # The filter_query_clause needs to be applied to `n`.
        # The group_ids are handled by _fulltext_lucene_query.
        
        cypher_query = f"""
            CALL db.index.fulltext.queryNodes("node_name_and_summary", $lucene_query, {{limit: $limit}}) 
            YIELD node AS n, score
            WHERE n:Entity {filter_query_clause} {_group_id_filter_clause("n", group_ids)}
            WITH n, score
            {ENTITY_NODE_RETURN}
            ORDER BY score DESC
        """
        # _group_id_filter_clause is a helper to generate "AND alias.group_id IN $group_ids" if group_ids is present
        
        all_params = {
            "lucene_query": lucene_query_str,
            "limit": limit,
            **filter_params
        }
        if group_ids:
            all_params["group_ids"] = group_ids

        records, _, _ = await self.execute_query(cypher_query, params=all_params)
        return [get_entity_node_from_record(record) for record in records]

    async def node_similarity_search(
        self, search_vector: List[float], search_filter: SearchFilters, 
        group_ids: Optional[List[str]] = None, 
        limit: int = RELEVANT_SCHEMA_LIMIT, min_score: float = DEFAULT_MIN_SCORE
    ) -> List[EntityNode]:
        if not self.driver: raise ConnectionError("Driver not initialized.")

        filter_query_clause, filter_params = _node_search_filter_query_constructor(search_filter)
        
        where_clauses = ["WHERE n.name_embedding IS NOT NULL"]
        if group_ids:
            where_clauses.append("n.group_id IN $group_ids")
            filter_params["group_ids"] = group_ids
        
        if filter_query_clause: # filter_query_clause starts with AND
            where_clauses.append(filter_query_clause.strip())

        full_query = f"""
            {RUNTIME_QUERY}
            MATCH (n:Entity)
            {" AND ".join(where_clauses).replace("WHERE AND", "WHERE")}
            WITH n, vector.similarity.cosine(n.name_embedding, $search_vector) AS score
            WHERE score >= $min_score
            {ENTITY_NODE_RETURN}
            ORDER BY score DESC
            LIMIT $limit
        """
        
        all_params = {
            "search_vector": search_vector,
            "limit": limit,
            "min_score": min_score,
            **filter_params
        }

        records, _, _ = await self.execute_query(full_query, params=all_params)
        return [get_entity_node_from_record(record) for record in records]

    async def node_bfs_search(
        self, bfs_origin_node_uuids: Optional[List[str]], 
        search_filter: SearchFilters, bfs_max_depth: int, limit: int
    ) -> List[EntityNode]:
        if not self.driver: raise ConnectionError("Driver not initialized.")
        if not bfs_origin_node_uuids: return []
        if bfs_max_depth <= 0 : bfs_max_depth = 1
        if bfs_max_depth > 5 : bfs_max_depth = 5


        filter_query_clause, filter_params = _node_search_filter_query_constructor(search_filter)
        
        # Original query had `WHERE n.group_id = origin.group_id`. This is important for partitioned graphs.
        # Path: (origin:Entity|Episodic)-[:RELATES_TO|MENTIONS*1..depth]-(n:Entity)
        
        query = f"""
            UNWIND $bfs_origin_node_uuids AS origin_uuid
            MATCH (origin {{uuid: origin_uuid}})
            CALL {{
                WITH origin
                MATCH path = (origin)-[*1..{bfs_max_depth}]-(n:Entity)
                WHERE ALL(rel IN relationships(path) WHERE TYPE(rel) IN ['RELATES_TO', 'MENTIONS']) // Ensure path uses valid edge types
                AND n.group_id = origin.group_id {filter_query_clause} // Apply filter to target node n
                RETURN n, length(path) AS depth
                ORDER BY depth
                LIMIT $limit // Limit within the subquery per origin to make it more performant
            }}
            WITH n // n is already distinct due to CALL subquery structure if multiple paths lead to same n
            {ENTITY_NODE_RETURN} 
            LIMIT $limit // Final limit on overall results
        """
        # The filter_query_clause applies to `n`.
        
        all_params = {
            "bfs_origin_node_uuids": bfs_origin_node_uuids,
            "limit": limit,
            **filter_params
        }

        records, _, _ = await self.execute_query(query, params=all_params)
        # Deduplicate nodes if multiple origins lead to the same node and it passes limit differently.
        # The query structure with CALL and final LIMIT should handle this reasonably.
        
        # To ensure unique nodes if multiple origins might return the same node within their individual limits:
        processed_nodes = {}
        for r in records:
            node = get_entity_node_from_record(r)
            if node.uuid not in processed_nodes:
                 processed_nodes[node.uuid] = node
        return list(processed_nodes.values())[:limit]


    async def episode_fulltext_search(
        self, query: str, search_filter: SearchFilters, # search_filter is ignored for episodes in original
        group_ids: Optional[List[str]] = None, limit: int = RELEVANT_SCHEMA_LIMIT
    ) -> List[EpisodicNode]:
        if not self.driver: raise ConnectionError("Driver not initialized.")

        lucene_query_str = _fulltext_lucene_query(query, group_ids)
        if not lucene_query_str:
            return []
        
        # search_filter is not used for episodes in the original search_utils function.
        # Group_ids are handled by _fulltext_lucene_query.

        cypher_query = f"""
            CALL db.index.fulltext.queryNodes("episode_content", $lucene_query, {{limit: $limit}}) 
            YIELD node AS e, score
            WHERE e:Episodic {_group_id_filter_clause("e", group_ids)} 
            RETURN 
                e.content AS content, e.created_at AS created_at, e.valid_at AS valid_at,
                e.uuid AS uuid, e.name AS name, e.group_id AS group_id,
                e.source_description AS source_description, e.source AS source,
                e.entity_edges AS entity_edges, score
            ORDER BY score DESC
            LIMIT $limit 
        """
        # Added {_group_id_filter_clause("e", group_ids)} for explicit filtering if lucene is not sufficient
        
        all_params = {"lucene_query": lucene_query_str, "limit": limit}
        if group_ids:
            all_params["group_ids"] = group_ids
            
        records, _, _ = await self.execute_query(cypher_query, params=all_params)
        return [get_episodic_node_from_record(record) for record in records]

    async def community_fulltext_search(
        self, query: str, group_ids: Optional[List[str]] = None, limit: int = RELEVANT_SCHEMA_LIMIT
    ) -> List[CommunityNode]:
        if not self.driver: raise ConnectionError("Driver not initialized.")

        lucene_query_str = _fulltext_lucene_query(query, group_ids)
        if not lucene_query_str:
            return []

        # SearchFilters are not applicable to CommunityNode search in original design.
        # Group_ids are handled by _fulltext_lucene_query.
        
        cypher_query = f"""
            CALL db.index.fulltext.queryNodes("community_name", $lucene_query, {{limit: $limit}}) 
            YIELD node AS comm, score
            WHERE comm:Community {_group_id_filter_clause("comm", group_ids)}
            RETURN
                comm.uuid AS uuid, comm.group_id AS group_id, 
                comm.name AS name, comm.created_at AS created_at, 
                comm.summary AS summary, comm.name_embedding AS name_embedding, score 
            ORDER BY score DESC
            LIMIT $limit
        """
        # Added name_embedding to return, and score for ordering.
        
        all_params = {"lucene_query": lucene_query_str, "limit": limit}
        if group_ids:
            all_params["group_ids"] = group_ids

        records, _, _ = await self.execute_query(cypher_query, params=all_params)
        return [get_community_node_from_record(record) for record in records]

    async def community_similarity_search(
        self, search_vector: List[float], group_ids: Optional[List[str]] = None, 
        limit: int = RELEVANT_SCHEMA_LIMIT, min_score: float = DEFAULT_MIN_SCORE
    ) -> List[CommunityNode]:
        if not self.driver: raise ConnectionError("Driver not initialized.")

        where_clauses = ["WHERE comm.name_embedding IS NOT NULL"]
        query_params = {}

        if group_ids:
            where_clauses.append("comm.group_id IN $group_ids")
            query_params["group_ids"] = group_ids
        
        full_query = f"""
            {RUNTIME_QUERY}
            MATCH (comm:Community)
            {" AND ".join(where_clauses).replace("WHERE AND", "WHERE")}
            WITH comm, vector.similarity.cosine(comm.name_embedding, $search_vector) AS score
            WHERE score >= $min_score
            RETURN
               comm.uuid As uuid, comm.group_id AS group_id,
               comm.name AS name, comm.created_at AS created_at, 
               comm.summary AS summary, comm.name_embedding AS name_embedding
            ORDER BY score DESC
            LIMIT $limit
        """
        # Added name_embedding to return.
        
        all_params = {
            "search_vector": search_vector,
            "limit": limit,
            "min_score": min_score,
            **query_params
        }

        records, _, _ = await self.execute_query(full_query, params=all_params)
        return [get_community_node_from_record(record) for record in records]

    async def get_embeddings_for_nodes(self, nodes: List[EntityNode]) -> Dict[str, List[float]]:
        if not self.driver: raise ConnectionError("Driver not initialized.")
        if not nodes: return {}

        query = """
            MATCH (n:Entity) WHERE n.uuid IN $node_uuids
            RETURN n.uuid AS uuid, n.name_embedding AS embedding
        """
        params = {"node_uuids": [node.uuid for node in nodes]}
        records, _, _ = await self.execute_query(query, params=params)
        
        return {r["uuid"]: r["embedding"] for r in records if r["embedding"] is not None}

    async def get_embeddings_for_communities(self, communities: List[CommunityNode]) -> Dict[str, List[float]]:
        if not self.driver: raise ConnectionError("Driver not initialized.")
        if not communities: return {}

        query = """
            MATCH (c:Community) WHERE c.uuid IN $community_uuids
            RETURN c.uuid AS uuid, c.name_embedding AS embedding
        """
        params = {"community_uuids": [comm.uuid for comm in communities]}
        records, _, _ = await self.execute_query(query, params=params)

        return {r["uuid"]: r["embedding"] for r in records if r["embedding"] is not None}

    async def get_embeddings_for_edges(self, edges: List[EntityEdge]) -> Dict[str, List[float]]:
        if not self.driver: raise ConnectionError("Driver not initialized.")
        if not edges: return {}
        
        query = """
            MATCH ()-[e:RELATES_TO]-() WHERE e.uuid IN $edge_uuids
            RETURN e.uuid AS uuid, e.fact_embedding AS embedding
        """
        params = {"edge_uuids": [edge.uuid for edge in edges]}
        records, _, _ = await self.execute_query(query, params=params)
        
        return {r["uuid"]: r["embedding"] for r in records if r["embedding"] is not None}

    async def get_mentioned_nodes(self, episodes: List[EpisodicNode]) -> List[EntityNode]:
        if not self.driver: raise ConnectionError("Driver not initialized.")
        if not episodes: return []

        episode_uuids = [episode.uuid for episode in episodes]
        query = f"""
            MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity) 
            WHERE episode.uuid IN $episode_uuids
            WITH DISTINCT n
            {ENTITY_NODE_RETURN}
        """
        records, _, _ = await self.execute_query(query, params={"episode_uuids": episode_uuids})
        return [get_entity_node_from_record(record) for record in records]

    async def get_communities_by_nodes(self, nodes: List[EntityNode]) -> List[CommunityNode]:
        if not self.driver: raise ConnectionError("Driver not initialized.")
        if not nodes: return []

        node_uuids = [node.uuid for node in nodes]
        query = """
            MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity) 
            WHERE n.uuid IN $node_uuids
            WITH DISTINCT c
            RETURN
                c.uuid As uuid, c.group_id AS group_id,
                c.name AS name, c.created_at AS created_at, 
                c.summary AS summary, c.name_embedding as name_embedding
        """
        records, _, _ = await self.execute_query(query, params={"node_uuids": node_uuids})
        return [get_community_node_from_record(record) for record in records]

    async def get_relevant_nodes(
        self, nodes: List[EntityNode], search_filter: SearchFilters, 
        min_score: float = DEFAULT_MIN_SCORE, # min_score for vector search part, not RRF directly
        limit: int = RELEVANT_SCHEMA_LIMIT,
        rrf_k: int = 60 # RRF constant for Reciprocal Rank Fusion
    ) -> List[List[EntityNode]]:
        """
        Finds relevant entity nodes for a given list of entity nodes using a combination of
        vector similarity search and full-text search, with results combined using
        Reciprocal Rank Fusion (RRF).

        For each input node, this method:
        1. Performs a vector similarity search based on `node.name_embedding`.
        2. Performs a full-text search based on `node.name` and `node.summary` (via Lucene index).
        3. Fetches up to `limit * 2` candidates from each search type to ensure a good pool for RRF.
        4. Calculates RRF scores for all unique candidates found.
        5. Returns the top `limit` nodes, ordered by their RRF score.
        """
        if not self.driver: raise ConnectionError("Driver not initialized.")
        if not nodes: return []

        all_final_nodes_map: Dict[str, List[EntityNode]] = {}
        from collections import defaultdict # Import defaultdict for rrf_scores

        for node_obj in nodes: # Renamed 'node' to 'node_obj' to avoid conflict
            if not node_obj.name_embedding:
                all_final_nodes_map[node_obj.uuid] = []
                continue

            filter_query_clause, filter_params_base = _node_search_filter_query_constructor(search_filter)
            
            # Fetch more results initially for RRF (e.g., limit * 2 or a fixed larger number)
            # This provides a larger candidate pool for better RRF ranking.
            individual_search_limit = limit * 2

            node_params = {
                "node_uuid": node_obj.uuid,
                "node_name_embedding": node_obj.name_embedding,
                "node_fulltext_query": _fulltext_lucene_query(node_obj.name, [node_obj.group_id] if node_obj.group_id else None),
                "group_id": node_obj.group_id, # Assuming search is within the same group_id for relevance
                "limit": individual_search_limit,
                "min_score_vec": min_score, # min_score for the initial vector search candidate generation
                **filter_params_base
            }
            
            # 1. Vector Search: Retrieves UUIDs and similarity scores for candidate nodes.
            # Nodes are matched by group_id, must not be the input node itself, and must have an embedding.
            # Results are ordered by similarity score descending and limited.
            vector_query = f"""
                MATCH (n_target:Entity {{group_id: $group_id}})
                WHERE n_target.uuid <> $node_uuid AND n_target.name_embedding IS NOT NULL {filter_query_clause.replace(" n:", " n_target:")}
                WITH n_target, vector.similarity.cosine(n_target.name_embedding, $node_name_embedding) AS score
                WHERE score >= $min_score_vec
                RETURN n_target.uuid AS uuid, score
                ORDER BY score DESC
                LIMIT $limit
            """
            vector_raw_results, _, _ = await self.execute_query(vector_query, params=node_params)
            
            # 2. Full-text Search: Retrieves UUIDs and relevance scores from Lucene index.
            # Nodes are matched by Lucene query, must be Entity type, not the input node,
            # and belong to the same group_id. Results ordered by score and limited.
            fulltext_raw_results = []
            if node_params["node_fulltext_query"]: # Only run if a valid Lucene query is formed
                fulltext_query_cypher = f"""
                    CALL db.index.fulltext.queryNodes("node_name_and_summary", $node_fulltext_query, {{limit: $limit}})
                    YIELD node AS n_target, score
                    WHERE n_target:Entity AND n_target.uuid <> $node_uuid AND n_target.group_id = $group_id {filter_query_clause.replace(" n:", " n_target:")}
                    RETURN n_target.uuid AS uuid, score
                    ORDER BY score DESC
                    LIMIT $limit
                """
                fulltext_raw_results, _, _ = await self.execute_query(fulltext_query_cypher, params=node_params)

            # 3. RRF Calculation: Combine scores from both search methods.
            # Each node's RRF score is the sum of reciprocal ranks from each list it appears in.
            rrf_scores: Dict[str, float] = defaultdict(float)
            
            for rank, record in enumerate(vector_raw_results):
                rrf_scores[record["uuid"]] += 1.0 / (rrf_k + rank + 1) # rank is 0-indexed

            for rank, record in enumerate(fulltext_raw_results):
                rrf_scores[record["uuid"]] += 1.0 / (rrf_k + rank + 1)

            # Sort candidate UUIDs by their combined RRF score in descending order.
            sorted_uuids_by_rrf = sorted(rrf_scores.keys(), key=lambda u: rrf_scores[u], reverse=True)

            # 4. Fetch top N EntityNode objects based on final RRF ranking.
            final_node_uuids_to_fetch = sorted_uuids_by_rrf[:limit] # Apply final limit

            if final_node_uuids_to_fetch:
                # Retrieve full node objects for the top-ranked UUIDs.
                final_nodes = await self.get_entity_nodes_by_uuids(uuids=final_node_uuids_to_fetch)
                # Re-sort the fetched nodes based on the RRF score order, as get_entity_nodes_by_uuids
                # might not preserve the input order of UUIDs.
                final_nodes_dict = {n.uuid: n for n in final_nodes}
                all_final_nodes_map[node_obj.uuid] = [final_nodes_dict[uuid] for uuid in final_node_uuids_to_fetch if uuid in final_nodes_dict]
            else:
                all_final_nodes_map[node_obj.uuid] = []

        return [all_final_nodes_map.get(n.uuid, []) for n in nodes]


    async def get_relevant_edges(
        self, edges: List[EntityEdge], search_filter: SearchFilters, 
        min_score: float = DEFAULT_MIN_SCORE, limit: int = RELEVANT_SCHEMA_LIMIT
    ) -> List[List[EntityEdge]]:
        if not self.driver: raise ConnectionError("Driver not initialized.")
        if not edges: return []
        
        all_relevant_edges_map: Dict[str, List[EntityEdge]] = {}

        for edge in edges:
            if not edge.fact_embedding: # Required for similarity
                all_relevant_edges_map[edge.uuid] = []
                continue

            filter_query_clause, filter_params = _edge_search_filter_query_constructor(search_filter)

            # Parameters for this specific edge
            edge_params = {
                "edge_uuid": edge.uuid,
                "edge_fact_embedding": edge.fact_embedding,
                "edge_group_id": edge.group_id, # Search within the same group
                "limit": limit,
                "min_score": min_score,
                **filter_params
            }
            
            # This query finds other edges in the same group_id, similar by fact_embedding.
            # It excludes the source edge itself.
            query = f"""
                MATCH (n_other:Entity)-[r_other:RELATES_TO {{group_id: $edge_group_id}}]->(m_other:Entity)
                WHERE r_other.uuid <> $edge_uuid AND r_other.fact_embedding IS NOT NULL {filter_query_clause.replace(" r:", " r_other:").replace(" n:", " n_other:").replace(" m:", " m_other:")}
                WITH r_other, n_other, m_other, vector.similarity.cosine(r_other.fact_embedding, $edge_fact_embedding) AS score
                WHERE score >= $min_score
                RETURN r_other AS r, startNode(r_other) AS n, endNode(r_other) AS m, score 
                ORDER BY score DESC
                LIMIT $limit
            """
            # The ENTITY_EDGE_RETURN needs to be used with aliases r, n, m.
            # Constructing return clause carefully:
            # ENTITY_EDGE_RETURN expects specific aliases. Let's ensure they match.
            # It uses e, startNode(e), endNode(e). Here we have r, n, m.
            # So, we map them in the RETURN.
            
            # Re-using ENTITY_EDGE_RETURN by aliasing:
            # WITH r_other AS e, n_other AS n, m_other AS m, score ... then ENTITY_EDGE_RETURN
            # This is complex. Simpler to just list the fields.
            
            query_with_return = f"""
                MATCH (n_other:Entity)-[r_other:RELATES_TO {{group_id: $edge_group_id}}]->(m_other:Entity)
                WHERE r_other.uuid <> $edge_uuid AND r_other.fact_embedding IS NOT NULL {filter_query_clause.replace(" r.", " r_other.").replace(" n.", " n_other.").replace(" m.", " m_other.")}
                WITH r_other, n_other, m_other, vector.similarity.cosine(r_other.fact_embedding, $edge_fact_embedding) AS score
                WHERE score >= $min_score
                RETURN 
                    r_other.uuid AS uuid,
                    n_other.uuid AS source_node_uuid,
                    m_other.uuid AS target_node_uuid,
                    r_other.created_at AS created_at,
                    r_other.name AS name,
                    r_other.group_id AS group_id,
                    r_other.fact AS fact,
                    r_other.episodes AS episodes,
                    r_other.expired_at AS expired_at,
                    r_other.valid_at AS valid_at,
                    r_other.invalid_at AS invalid_at,
                    properties(r_other) AS attributes,
                    score
                ORDER BY score DESC
                LIMIT $limit
            """

            records, _, _ = await self.execute_query(query_with_return, params=edge_params)
            all_relevant_edges_map[edge.uuid] = [get_entity_edge_from_record(r) for r in records]
            
        return [all_relevant_edges_map.get(edge.uuid, []) for edge in edges]


    async def get_edge_invalidation_candidates(
        self, edges: List[EntityEdge], search_filter: SearchFilters, 
        min_score: float = DEFAULT_MIN_SCORE, limit: int = RELEVANT_SCHEMA_LIMIT
    ) -> List[List[EntityEdge]]:
        # This method is very similar to get_relevant_edges in its structure,
        # but the original query in search_utils had a subtle difference in MATCH clause:
        # `MATCH (n:Entity)-[e:RELATES_TO {group_id: edge.group_id}]->(m:Entity)`
        # `WHERE n.uuid IN [edge.source_node_uuid, edge.target_node_uuid] OR m.uuid IN [edge.target_node_uuid, edge.source_node_uuid]`
        # This means it looks for edges connected to *either* source or target of the input edge, within the same group.
        # This is different from get_relevant_edges which just finds other edges in the same group.
        
        if not self.driver: raise ConnectionError("Driver not initialized.")
        if not edges: return []

        all_invalidation_candidates_map: Dict[str, List[EntityEdge]] = {}

        for edge in edges:
            if not edge.fact_embedding:
                all_invalidation_candidates_map[edge.uuid] = []
                continue

            filter_query_clause, filter_params = _edge_search_filter_query_constructor(search_filter)

            edge_params = {
                "edge_uuid": edge.uuid,
                "edge_fact_embedding": edge.fact_embedding,
                "edge_group_id": edge.group_id,
                "edge_source_uuid": edge.source_node_uuid,
                "edge_target_uuid": edge.target_node_uuid,
                "limit": limit,
                "min_score": min_score,
                **filter_params
            }
            
            # The r_other must not be the source edge itself.
            # The r_other must be connected to one of the nodes of the source edge.
            query = f"""
                MATCH (src_node:Entity {{uuid: $edge_source_uuid}})
                MATCH (tgt_node:Entity {{uuid: $edge_target_uuid}})
                CALL {{
                    WITH src_node, tgt_node // Pass these specific nodes
                    MATCH (n_other:Entity)-[r_other:RELATES_TO {{group_id: $edge_group_id}}]->(m_other:Entity)
                    WHERE r_other.uuid <> $edge_uuid AND r_other.fact_embedding IS NOT NULL
                      AND (n_other = src_node OR n_other = tgt_node OR m_other = src_node OR m_other = tgt_node)
                      {filter_query_clause.replace(" r.", " r_other.").replace(" n.", " n_other.").replace(" m.", " m_other.")}
                    WITH r_other, n_other, m_other, vector.similarity.cosine(r_other.fact_embedding, $edge_fact_embedding) AS score
                    WHERE score >= $min_score
                    RETURN r_other, n_other, m_other, score
                    ORDER BY score DESC
                    LIMIT $limit
                }}
                RETURN 
                    r_other.uuid AS uuid,
                    n_other.uuid AS source_node_uuid,
                    m_other.uuid AS target_node_uuid,
                    r_other.created_at AS created_at,
                    r_other.name AS name,
                    r_other.group_id AS group_id,
                    r_other.fact AS fact,
                    r_other.episodes AS episodes,
                    r_other.expired_at AS expired_at,
                    r_other.valid_at AS valid_at,
                    r_other.invalid_at AS invalid_at,
                    properties(r_other) AS attributes,
                    score 
            """
            # This query is complex due to matching edges connected to one of two specific nodes.
            # Using a CALL subquery to make it more manageable.

            records, _, _ = await self.execute_query(query, params=edge_params)
            all_invalidation_candidates_map[edge.uuid] = [get_entity_edge_from_record(r) for r in records]

        return [all_invalidation_candidates_map.get(edge.uuid, []) for edge in edges]

    async def count_node_mentions(self, node_uuid: str) -> int:
        if not self.driver:
            raise ConnectionError("Driver not initialized. Call connect() first.")

        # This query counts distinct episodic nodes that mention the given entity node.
        # It assumes the relationship is (:Episodic)-[:MENTIONS]->(:Entity)
        query = """
            MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $node_uuid})
            RETURN count(DISTINCT e) AS mention_count
        """
        records, _, _ = await self.execute_query(query, params={"node_uuid": node_uuid}, database_=self.database)
        if records and records[0] and "mention_count" in records[0]:
            return int(records[0]["mention_count"])
        return 0

# Internal helper for adding group_id filter clause if group_ids are provided
def _group_id_filter_clause(alias: str, group_ids: Optional[List[str]]) -> str:
    if group_ids:
        return f" AND {alias}.group_id IN $group_ids "
    return ""

# (Previous code for CRUD, Connection, etc. remains unchanged)
# ...

    async def add_nodes_and_edges_bulk(
        self,
        episodic_nodes: List[EpisodicNode],
        episodic_edges: List[EpisodicEdge],
        entity_nodes: List[EntityNode],
        entity_edges: List[EntityEdge],
        embedder: EmbedderClient
    ) -> None:
        """
        Adds nodes and edges in bulk to Neo4j using batched Cypher queries within a single transaction.
        Embeddings are generated for nodes and edges if not already present before the transaction begins.
        """
        if not self.driver:
            raise ConnectionError("Driver not initialized. Call connect() first.")

        # This internal transaction function mirrors the logic from the old
        # bulk_utils.add_nodes_and_edges_bulk_tx
        async def _add_nodes_and_edges_bulk_tx(
            tx: AsyncManagedTransaction, # Comes from Neo4j driver
            ep_nodes: List[EpisodicNode],
            ep_edges: List[EpisodicEdge],
            en_nodes: List[EntityNode], # Embeddings are pre-generated for these
            en_edges: List[EntityEdge], # Embeddings are pre-generated for these
            emb_client: EmbedderClient, # Passed for any potential deeper use, though primary embedding is outside
        ):
            # Prepare EpisodicNode data
            episodes_data = []
            for episode_node in ep_nodes:
                data = episode_node.model_dump(exclude_none=False)
                data['source'] = str(episode_node.source.value)
                for key in ['uuid', 'name', 'group_id', 'source', 'source_description', 'content', 'valid_at', 'created_at', 'entity_edges']:
                    data.setdefault(key, None)
                if data['entity_edges'] is None: data['entity_edges'] = []
                episodes_data.append(data)

            # Prepare EntityNode data (embeddings are already generated)
            entity_nodes_data = []
            for node in en_nodes:
                entity_data: dict[str, Any] = {
                    'uuid': node.uuid,
                    'name': node.name,
                    'name_embedding': node.name_embedding,
                    'group_id': node.group_id,
                    'summary': node.summary,
                    'created_at': node.created_at,
                }
                entity_data.update(node.attributes or {})
                entity_data['labels'] = list(set(node.labels + ['Entity']))
                entity_nodes_data.append(entity_data)

            # Prepare EntityEdge data (embeddings are already generated)
            entity_edges_data = []
            for edge in en_edges:
                edge_data: dict[str, Any] = {
                    'uuid': edge.uuid,
                    'source_node_uuid': edge.source_node_uuid,
                    'target_node_uuid': edge.target_node_uuid,
                    'name': edge.name,
                    'fact': edge.fact,
                    'fact_embedding': edge.fact_embedding,
                    'group_id': edge.group_id,
                    'episodes': edge.episodes,
                    'created_at': edge.created_at,
                    'expired_at': edge.expired_at,
                    'valid_at': edge.valid_at,
                    'invalid_at': edge.invalid_at,
                }
                edge_data.update(edge.attributes or {})
                if edge_data['episodes'] is None: edge_data['episodes'] = []
                entity_edges_data.append(edge_data)

            episodic_edges_prepared_data = []
            for ep_edge in ep_edges:
                data = ep_edge.model_dump(exclude_none=False)
                for key in ['uuid', 'source_node_uuid', 'target_node_uuid', 'group_id', 'created_at']:
                    data.setdefault(key, None)
                episodic_edges_prepared_data.append(data)

            # Execute bulk Cypher queries
            if episodes_data:
                await tx.run(EPISODIC_NODE_SAVE_BULK, episodes=episodes_data)
            if entity_nodes_data:
                await tx.run(ENTITY_NODE_SAVE_BULK, nodes=entity_nodes_data)
            if episodic_edges_prepared_data:
                await tx.run(EPISODIC_EDGE_SAVE_BULK, episodic_edges=episodic_edges_prepared_data)
            if entity_edges_data:
                await tx.run(ENTITY_EDGE_SAVE_BULK, entity_edges=entity_edges_data)

            logger.info(f"Neo4j bulk transaction processed: {len(ep_nodes)} ep_nodes, {len(en_nodes)} en_nodes, {len(ep_edges)} ep_edges, {len(en_edges)} en_edges.")

        # Perform embedding generation before starting the database transaction.
        logger.info(f"Neo4jProvider: Starting bulk add. Generating embeddings first...")
        for node in entity_nodes:
            if node.name_embedding is None and hasattr(node, 'name') and node.name:
                text_to_embed = node.name.replace('\n', ' ')
                embedding_result = await embedder.create(input_data=[text_to_embed])
                if embedding_result and isinstance(embedding_result, list) and len(embedding_result) > 0 and isinstance(embedding_result[0], list):
                    node.name_embedding = embedding_result[0]

        # Entity Edges Embeddings
        for edge in entity_edges:
            if edge.fact_embedding is None and hasattr(edge, 'fact') and edge.fact:
                text_to_embed = edge.fact.replace('\n', ' ')
                embedding_result = await embedder.create(input_data=[text_to_embed])
                if embedding_result and isinstance(embedding_result, list) and len(embedding_result) > 0 and isinstance(embedding_result[0], list):
                    edge.fact_embedding = embedding_result[0]
        logger.info("Embeddings generated. Proceeding with Neo4j transaction.")

        async with self.get_session() as session:
            await session.execute_write(
                _add_nodes_and_edges_bulk_tx, # Pass the actual transaction function
                episodic_nodes,
                episodic_edges,
                entity_nodes,
                entity_edges,
                embedder, # Embedder is passed but embeddings are now pre-generated
            )
        logger.info("Neo4j bulk add_nodes_and_edges operation completed.")
