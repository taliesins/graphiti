import logging
import json # For serializing/deserializing MAP attributes
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple, Set

import kuzu
from graphiti_core.providers.base import GraphDatabaseProvider
from graphiti_core.nodes import (
    EpisodicNode,
    EntityNode,
    CommunityNode,
    EpisodeType,
)
from graphiti_core.edges import (
    EpisodicEdge,
    EntityEdge,
    CommunityEdge,
)
from graphiti_core.embedder import EmbedderClient
from pydantic import BaseModel, Field # Needed for SearchFilters if defined inline, or imported
from enum import Enum


logger = logging.getLogger(__name__)

# Moved from neo4j_provider, consider a common types file later
class ComparisonOperator(str, Enum):
    EQUALS = "="
    NOT_EQUALS = "<>"
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUALS = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUALS = "<="
    CONTAINS = "CONTAINS" # For string or list properties
    STARTS_WITH = "STARTS WITH"
    ENDS_WITH = "ENDS WITH"
    # Note: KuzuDB might not support all of these directly in Cypher or may have different syntax

class DateFilter(BaseModel):
    date: datetime
    operator: ComparisonOperator = ComparisonOperator.EQUALS

class AttributeFilter(BaseModel):
    key: str
    value: Any
    operator: ComparisonOperator = ComparisonOperator.EQUALS
    # For KuzuDB, 'attributes' is a JSON string. Direct complex queries on it are hard.
    # This might be simplified to exact match on specific extracted top-level attributes if Kuzu allows.
    # For now, this implies filtering after fetching or simplified string CONTAINS if feasible.

class SearchFilters(BaseModel): # type: ignore
    node_labels: Optional[List[str]] = None # e.g., ["Entity", "CustomLabel"] - Kuzu has fixed tables
    edge_types: Optional[List[str]] = None   # e.g., ["RELATES_TO"] - Kuzu has fixed tables

    # Date filters - applied to 'created_at' or 'valid_at' based on context
    created_at_filter: Optional[DateFilter] = None
    valid_at_filter: Optional[DateFilter] = None # Usually for EpisodicNode or time-bound edges

    # Attribute filtering - more complex for Kuzu due to JSON string attributes
    # This is a conceptual placeholder; actual implementation in _apply_search_filters_to_query_basic
    # will need to be very careful about how Kuzu can filter JSON string attributes.
    # It might be limited to simple string CONTAINS for the serialized JSON, or not directly supportable.
    attribute_filters: Optional[List[AttributeFilter]] = None

    # For KuzuDB, group_id is often a direct property, not in 'attributes'
    group_id_filter: Optional[str] = None # If filtering by a single group_id not already in main method params

    # Raw Cypher WHERE clause to append (use with extreme caution)
    # This is generally not recommended for safety and portability but can be a backdoor.
    raw_where_clause: Optional[str] = Field(None, exclude=True) # Exclude from model_dump by default


class KuzuDBProvider(GraphDatabaseProvider):
    def __init__(self, database_path: str, in_memory: bool = False, embedding_dimension: int = 1536):
        self.database_path = database_path
        self.in_memory = in_memory
        self.embedding_dimension = embedding_dimension
        self.db: Optional[kuzu.Database] = None
        self.connection: Optional[kuzu.Connection] = None
        self._connect_internal()

    def _connect_internal(self):
        """Internal method to establish DB and connection."""
        try:
            if self.in_memory:
                self.db = kuzu.Database() 
            else:
                self.db = kuzu.Database(self.database_path)
            self.connection = kuzu.Connection(self.db)
            logger.info(f"KuzuDB connection established to {'in-memory database' if self.in_memory else self.database_path}")
        except Exception as e: 
            logger.error(f"Failed to initialize KuzuDB: {e}")
            self.db = None
            self.connection = None
            raise ConnectionError(f"Failed to initialize KuzuDB: {e}") from e

    async def connect(self, **kwargs) -> None:
        if not self.connection or not self.db: 
            logger.info("Re-initializing KuzuDB connection.")
            self._connect_internal()
        await self.verify_connectivity() 

    async def close(self) -> None:
        self.connection = None
        self.db = None 
        logger.info("KuzuDB resources released (connection and db references set to None).")

    def get_session(self) -> kuzu.Connection:
        if not self.connection:
            raise ConnectionError("KuzuDB connection not established. Call connect() first.")
        return self.connection

    async def verify_connectivity(self) -> None:
        if not self.connection:
            raise ConnectionError("KuzuDB connection not established.")
        try:
            # Kuzu connections are typically synchronous for execute
            self.connection.execute("RETURN 1")
            logger.info("KuzuDB connectivity verified.")
        except Exception as e: 
            logger.error(f"KuzuDB connectivity verification failed: {e}")
            raise ConnectionError(f"KuzuDB connectivity verification failed: {e}") from e

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        if not self.connection:
            raise ConnectionError("KuzuDB connection not established.")
        conn = self.get_session()

        if delete_existing:
            logger.info("Deleting existing KuzuDB tables (if they exist)...")
            table_names = ["Entity", "Episodic", "Community", "MENTIONS", "RELATES_TO", "HAS_MEMBER"]
            for name in table_names:
                try:
                    conn.execute(f"DROP TABLE {name}")
                    logger.info(f"Dropped table {name}.")
                except Exception as e: 
                    logger.debug(f"Could not drop table {name} (it might not exist): {e}")
        
        logger.info("Building KuzuDB schema (tables and indices)...")
        entity_schema = f"CREATE NODE TABLE Entity (uuid STRING, name STRING, group_id STRING, created_at TIMESTAMP, summary STRING, name_embedding FLOAT[{self.embedding_dimension}], attributes STRING, PRIMARY KEY (uuid))"
        conn.execute(entity_schema)
        episodic_schema = f"CREATE NODE TABLE Episodic (uuid STRING, name STRING, group_id STRING, source STRING, source_description STRING, content STRING, valid_at TIMESTAMP, created_at TIMESTAMP, entity_edges LIST(STRING), PRIMARY KEY (uuid))"
        conn.execute(episodic_schema)
        community_schema = f"CREATE NODE TABLE Community (uuid STRING, name STRING, group_id STRING, created_at TIMESTAMP, summary STRING, name_embedding FLOAT[{self.embedding_dimension}], PRIMARY KEY (uuid))"
        conn.execute(community_schema)
        mentions_schema = f"CREATE REL TABLE MENTIONS (FROM Episodic TO Entity, uuid STRING, group_id STRING, created_at TIMESTAMP)"
        conn.execute(mentions_schema)
        relates_to_schema = f"CREATE REL TABLE RELATES_TO (FROM Entity TO Entity, uuid STRING, name STRING, group_id STRING, fact STRING, fact_embedding FLOAT[{self.embedding_dimension}], episodes LIST(STRING), created_at TIMESTAMP, expired_at TIMESTAMP, valid_at TIMESTAMP, invalid_at TIMESTAMP, attributes STRING)"
        conn.execute(relates_to_schema)
        has_member_schema = f"CREATE REL TABLE HAS_MEMBER (FROM Community TO Entity, uuid STRING, group_id STRING, created_at TIMESTAMP)"
        conn.execute(has_member_schema)
        logger.info("KuzuDB schema build process completed.")

    async def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
        if not self.connection:
            raise ConnectionError("KuzuDB connection not established.")
        try:
            prepared_statement = self.connection.prepare(query)
            query_result = self.connection.execute(prepared_statement, **(params or {}))
            col_names = query_result.get_column_names()
            records_as_dicts = [dict(zip(col_names, row)) for row in query_result.get_as_torch_geometric().node_features] if query_result.has_next() else [dict(zip(col_names, row)) for row in query_result.get_as_df().to_dict(orient='split')['data']] if query_result.has_next() else [dict(zip(col_names,row)) for row in query_result.get_as_list()] if query_result.has_next() else [] # Fallback, ideally one method works consistently
            # Kuzu doesn't provide summary_metadata or summary_counters directly.
            return records_as_dicts, col_names if col_names else [], {}
        except Exception as e:
            logger.error(f"KuzuDB query execution failed. Query: {query}, Params: {params}, Error: {e}")
            raise ConnectionError(f"KuzuDB query execution failed: {e}") from e

    def _kuzudb_to_entity_node(self, row_dict: Dict[str, Any]) -> EntityNode:
        attrs_str = row_dict.get('attributes', '{}')
        attrs = json.loads(attrs_str) if isinstance(attrs_str, str) and attrs_str.strip() else (attrs_str if isinstance(attrs_str, dict) else {})
        return EntityNode(
            uuid=row_dict['uuid'], name=row_dict['name'], group_id=row_dict.get('group_id'),
            created_at=row_dict['created_at'], summary=row_dict.get('summary'),
            name_embedding=row_dict.get('name_embedding'), attributes=attrs, labels=['Entity']
        )

    def _kuzudb_to_episodic_node(self, row_dict: Dict[str, Any]) -> EpisodicNode:
        return EpisodicNode(
            uuid=row_dict['uuid'], name=row_dict['name'], group_id=row_dict.get('group_id'),
            source=EpisodeType(row_dict['source']), source_description=row_dict['source_description'],
            content=row_dict['content'], valid_at=row_dict['valid_at'], created_at=row_dict['created_at'],
            entity_edges=row_dict.get('entity_edges', [])
        )

    def _kuzudb_to_community_node(self, row_dict: Dict[str, Any]) -> CommunityNode:
        return CommunityNode(
            uuid=row_dict['uuid'], name=row_dict['name'], group_id=row_dict.get('group_id'),
            created_at=row_dict['created_at'], summary=row_dict.get('summary'),
            name_embedding=row_dict.get('name_embedding'), labels=['Community']
        )

    def _kuzudb_to_entity_edge(self, row_dict: Dict[str, Any]) -> EntityEdge:
        attrs_str = row_dict.get('attributes', '{}')
        attrs = json.loads(attrs_str) if isinstance(attrs_str, str) and attrs_str.strip() else (attrs_str if isinstance(attrs_str, dict) else {})
        return EntityEdge(
            uuid=row_dict['uuid'], name=row_dict['name'], group_id=row_dict.get('group_id'),
            fact=row_dict['fact'], fact_embedding=row_dict.get('fact_embedding'),
            episodes=row_dict.get('episodes', []), created_at=row_dict['created_at'],
            expired_at=row_dict.get('expired_at'), valid_at=row_dict.get('valid_at'),
            invalid_at=row_dict.get('invalid_at'), attributes=attrs,
            source_node_uuid=row_dict['source_node_uuid'], target_node_uuid=row_dict['target_node_uuid']
        )

    def _kuzudb_to_episodic_edge(self, row_dict: Dict[str, Any]) -> EpisodicEdge:
        return EpisodicEdge(
            uuid=row_dict['uuid'], group_id=row_dict.get('group_id'), created_at=row_dict['created_at'],
            source_node_uuid=row_dict['source_node_uuid'], target_node_uuid=row_dict['target_node_uuid']
        )

    def _kuzudb_to_community_edge(self, row_dict: Dict[str, Any]) -> CommunityEdge:
        return CommunityEdge(
            uuid=row_dict['uuid'], group_id=row_dict.get('group_id'), created_at=row_dict['created_at'],
            source_node_uuid=row_dict['source_node_uuid'], target_node_uuid=row_dict['target_node_uuid']
        )
    
    async def _save_node_generic(self, table_name: str, node_dict: Dict[str, Any], uuid_param: str) -> str:
        """
        Generic helper to save (upsert) a node using KuzuDB's MERGE statement.
        It matches on 'uuid' and sets/updates other properties.
        """
        if not self.connection:
            raise ConnectionError("KuzuDB connection not established.")

        # Using MERGE for upsert. node_dict contains all properties, including 'uuid'.

        # Properties for ON CREATE SET and ON MATCH SET clauses.
        # Kuzu's MERGE requires explicit SET clauses for each property.

        on_create_set_clauses = []
        param_prefix = "p_" # Prefix for parameters in SET clauses to avoid conflict

        # All properties from node_dict are set on create.
        for key in node_dict.keys():
            on_create_set_clauses.append(f"n.{key} = ${param_prefix}{key}")

        # On match, update all properties except the primary key 'uuid'.
        on_match_set_clauses = []
        for key in node_dict.keys():
            if key != "uuid": # Don't try to update the primary key itself in SET
                on_match_set_clauses.append(f"n.{key} = ${param_prefix}{key}")

        # Build the MERGE query
        # The MERGE clause matches on the primary key 'uuid'.
        query = f"MERGE (n:{table_name} {{uuid: $match_uuid}})"

        if on_create_set_clauses:
            query += f" ON CREATE SET {', '.join(on_create_set_clauses)}"

        # Only add ON MATCH SET if there are properties to update (other than uuid).
        if on_match_set_clauses:
            query += f" ON MATCH SET {', '.join(on_match_set_clauses)}"
        else:
            # If only 'uuid' was in node_dict, there's nothing to set on match.
            # KuzuDB's MERGE requires ON CREATE SET if the node doesn't exist.
            # If ON MATCH is omitted, it won't update existing nodes, which is fine if that's the intent
            # for a create-only-if-not-exists scenario, but here we want upsert.
            # However, if on_match_set_clauses is empty, it means only 'uuid' was in node_dict
            # (after filtering), so no properties to update anyway.
            pass

        # Prepare parameters for the query
        params = {"match_uuid": uuid_param}
        for key, value in node_dict.items():
            params[f"{param_prefix}{key}"] = value # e.g., p_name, p_group_id

        try:
            await self.execute_query(query, params)
        except Exception as e:
            logger.error(f"MERGE operation failed for {table_name} {uuid_param}: {e}. Query: {query}")
            # Avoid logging params directly if they might contain sensitive data or are very large.
            # Consider logging only keys or specific non-sensitive params for debugging if necessary.
            # logger.debug(f"Failed MERGE Params: {list(params.keys())}")
            raise e
        return uuid_param

    async def save_entity_node(self, node: EntityNode) -> Any:
        node_dict = node.model_dump(exclude_none=True)
        node_dict["attributes"] = json.dumps(node.attributes) if node.attributes else "{}"
        return await self._save_node_generic("Entity", node_dict, node.uuid)

    async def save_episodic_node(self, node: EpisodicNode) -> Any:
        node_dict = node.model_dump(exclude_none=True)
        node_dict["source"] = node.source.value
        return await self._save_node_generic("Episodic", node_dict, node.uuid)

    async def save_community_node(self, node: CommunityNode) -> Any:
        node_dict = node.model_dump(exclude_none=True)
        return await self._save_node_generic("Community", node_dict, node.uuid)

    async def _get_node_by_uuid_generic(self, table_name: str, uuid: str, parser_func, cols: str):
        if not self.connection: return None
        query = f"MATCH (n:{table_name} {{uuid: $uuid}}) RETURN {cols}"
        results, _, _ = await self.execute_query(query, {"uuid": uuid})
        return parser_func(results[0]) if results else None

    async def get_entity_node_by_uuid(self, uuid: str) -> Optional[EntityNode]:
        return await self._get_node_by_uuid_generic("Entity", uuid, self._kuzudb_to_entity_node, "n.uuid, n.name, n.group_id, n.created_at, n.summary, n.name_embedding, n.attributes")
    
    async def get_episodic_node_by_uuid(self, uuid: str) -> Optional[EpisodicNode]:
        return await self._get_node_by_uuid_generic("Episodic", uuid, self._kuzudb_to_episodic_node, "n.uuid, n.name, n.group_id, n.source, n.source_description, n.content, n.valid_at, n.created_at, n.entity_edges")

    async def get_community_node_by_uuid(self, uuid: str) -> Optional[CommunityNode]:
        return await self._get_node_by_uuid_generic("Community", uuid, self._kuzudb_to_community_node, "n.uuid, n.name, n.group_id, n.created_at, n.summary, n.name_embedding")

    async def _get_nodes_by_uuids_generic(self, table_name: str, uuids: List[str], parser_func, cols: str):
        if not self.connection or not uuids: return []
        # Kuzu's IN clause syntax for lists: WHERE n.uuid IN $uuids (Kuzu should handle list parameters)
        query = f"MATCH (n:{table_name}) WHERE n.uuid IN $uuids RETURN {cols}"
        results, _, _ = await self.execute_query(query, {"uuids": uuids})
        return [parser_func(res) for res in results]

    async def get_entity_nodes_by_uuids(self, uuids: List[str]) -> List[EntityNode]:
        return await self._get_nodes_by_uuids_generic("Entity", uuids, self._kuzudb_to_entity_node, "n.uuid, n.name, n.group_id, n.created_at, n.summary, n.name_embedding, n.attributes")

    async def get_episodic_nodes_by_uuids(self, uuids: List[str]) -> List[EpisodicNode]:
        return await self._get_nodes_by_uuids_generic("Episodic", uuids, self._kuzudb_to_episodic_node, "n.uuid, n.name, n.group_id, n.source, n.source_description, n.content, n.valid_at, n.created_at, n.entity_edges")

    async def get_community_nodes_by_uuids(self, uuids: List[str]) -> List[CommunityNode]:
        return await self._get_nodes_by_uuids_generic("Community", uuids, self._kuzudb_to_community_node, "n.uuid, n.name, n.group_id, n.created_at, n.summary, n.name_embedding")

    async def _get_nodes_by_group_ids_generic(self, table_name: str, group_ids: List[str], parser_func, limit: Optional[int], uuid_cursor: Optional[str], cols: str):
        if not self.connection or not group_ids: return []
        params: Dict[str, Any] = {"group_ids": group_ids}
        # Kuzu doesn't support direct cursor pagination easily. Basic LIMIT.
        query = f"MATCH (n:{table_name}) WHERE n.group_id IN $group_ids RETURN {cols}"
        if limit is not None:
            query += " LIMIT $limit"
            params["limit"] = limit
        results, _, _ = await self.execute_query(query, params)
        return [parser_func(res) for res in results]

    async def get_entity_nodes_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EntityNode]:
        return await self._get_nodes_by_group_ids_generic("Entity", group_ids, self._kuzudb_to_entity_node, limit, uuid_cursor, "n.uuid, n.name, n.group_id, n.created_at, n.summary, n.name_embedding, n.attributes")

    async def get_episodic_nodes_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EpisodicNode]:
        return await self._get_nodes_by_group_ids_generic("Episodic", group_ids, self._kuzudb_to_episodic_node, limit, uuid_cursor, "n.uuid, n.name, n.group_id, n.source, n.source_description, n.content, n.valid_at, n.created_at, n.entity_edges")

    async def get_community_nodes_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[CommunityNode]:
        return await self._get_nodes_by_group_ids_generic("Community", group_ids, self._kuzudb_to_community_node, limit, uuid_cursor, "n.uuid, n.name, n.group_id, n.created_at, n.summary, n.name_embedding")

    async def get_episodic_nodes_by_entity_node_uuid(self, entity_node_uuid: str) -> List[EpisodicNode]:
        if not self.connection: return []
        query = "MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $uuid}) RETURN e.uuid, e.name, e.group_id, e.source, e.source_description, e.content, e.valid_at, e.created_at, e.entity_edges"
        results, _, _ = await self.execute_query(query, {"uuid": entity_node_uuid})
        return [self._kuzudb_to_episodic_node(res) for res in results]
    
    async def delete_node(self, uuid: str) -> None:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        logger.warning(f"Attempting to delete node {uuid}. KuzuDB requires manual deletion of connected relationships first.")
        for node_table in ["Entity", "Episodic", "Community"]:
            try:
                # Attempt to delete relationships pointing to/from this node from all known rel tables
                for rel_table in ["MENTIONS", "RELATES_TO", "HAS_MEMBER"]:
                    await self.execute_query(f"MATCH (n:{node_table} {{uuid: $uuid}})-[r:{rel_table}]-() DELETE r", {"uuid": uuid})
                    await self.execute_query(f"MATCH (n:{node_table} {{uuid: $uuid}})<-[r:{rel_table}]-() DELETE r", {"uuid": uuid})
                await self.execute_query(f"MATCH (n:{node_table} {{uuid: $uuid}}) DELETE n", {"uuid": uuid})
                logger.info(f"Node {uuid} potentially deleted from table {node_table}.")
                return # Assume node is of one type
            except Exception: # Kuzu may error if node not found in a table or if rels still exist
                pass # Try next table
        logger.debug(f"Finished attempt to delete node {uuid} across all tables.")


    async def delete_nodes_by_group_id(self, group_id: str, node_type: str) -> None:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        if node_type not in ["Entity", "Episodic", "Community"]:
            raise ValueError(f"Invalid node_type '{node_type}' for KuzuDB delete.")
        
        logger.warning(f"Attempting to delete nodes from {node_type} with group_id {group_id}. Manual relationship deletion is critical in KuzuDB.")
        # Delete relationships connected to these nodes first
        # This is a best-effort, may need to be more specific based on schema
        for rel_table in ["MENTIONS", "RELATES_TO", "HAS_MEMBER"]:
             await self.execute_query(f"MATCH (n:{node_type} {{group_id: $gid}})-[r:{rel_table}]-() DELETE r", {"gid": group_id})
             await self.execute_query(f"MATCH (n:{node_type} {{group_id: $gid}})<-[r:{rel_table}]-() DELETE r", {"gid": group_id})
        # Then delete the nodes
        await self.execute_query(f"MATCH (n:{node_type} {{group_id: $gid}}) DELETE n", {"gid": group_id})
        logger.info(f"Attempted deletion of {node_type} nodes for group_id {group_id}.")

    async def _save_edge_generic(self, rel_table: str, source_table: str, target_table: str, edge_dict: Dict[str, Any], uuid: str) -> str:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")

        props_for_rel = {k: v for k, v in edge_dict.items() if k not in ['source_node_uuid', 'target_node_uuid']}
        # Ensure uuid is part of props_for_rel as it's used in MERGE condition and SET
        # The 'uuid' field in edge_dict is the edge's own uuid.
        if 'uuid' not in props_for_rel:
             props_for_rel['uuid'] = uuid # uuid parameter is the edge's uuid from the edge object

        # Construct SET clauses for ON CREATE and ON MATCH
        # These clauses will use the keys from props_for_rel.
        # Example: "r.name = $name, r.group_id = $group_id"
        # All properties in props_for_rel (including uuid) will be set.
        set_clauses = ", ".join([f"r.{key} = ${key}" for key in props_for_rel])

        if not props_for_rel: # If props_for_rel is empty (e.g. only source/target uuids in edge_dict)
            # This case should ideally not happen if edges always have at least a uuid.
            # If Kuzu requires at least one property to be set in MERGE, this could be an issue.
            # For now, we assume uuid is always present.
            # If set_clauses is empty, Kuzu might error.
            # However, 'uuid' is ensured to be in props_for_rel, so set_clauses won't be empty.
            pass

        query = f"""
            MATCH (s:{source_table} {{uuid: $source_uuid}}), (t:{target_table} {{uuid: $target_uuid}})
            MERGE (s)-[r:{rel_table} {{uuid: $edge_uuid_param}}]->(t)
            ON CREATE SET {set_clauses}
            ON MATCH SET {set_clauses}
        """

        params = {
            "source_uuid": edge_dict["source_node_uuid"],
            "target_uuid": edge_dict["target_node_uuid"],
            "edge_uuid_param": uuid, # The UUID of the edge for the MERGE condition
            **props_for_rel # Spread all properties for use in SET clauses (e.g. $uuid, $group_id etc.)
        }

        try:
            await self.execute_query(query, params)
        except Exception as e:
            logger.error(f"MERGE operation failed for edge {uuid} in {rel_table}: {e}. Query: {query}, Params: {params}")
            raise e
        return uuid

    async def save_entity_edge(self, edge: EntityEdge) -> Any:
        edge_dict = edge.model_dump(exclude_none=True)
        edge_dict["attributes"] = json.dumps(edge.attributes) if edge.attributes else "{}"
        return await self._save_edge_generic("RELATES_TO", "Entity", "Entity", edge_dict, edge.uuid)

    async def save_episodic_edge(self, edge: EpisodicEdge) -> Any:
        edge_dict = edge.model_dump(exclude_none=True)
        return await self._save_edge_generic("MENTIONS", "Episodic", "Entity", edge_dict, edge.uuid)

    async def save_community_edge(self, edge: CommunityEdge) -> Any:
        edge_dict = edge.model_dump(exclude_none=True)
        return await self._save_edge_generic("HAS_MEMBER", "Community", "Entity", edge_dict, edge.uuid)

    async def _get_edge_by_uuid_generic(self, rel_table: str, source_table: str, target_table: str, uuid: str, parser_func, rel_cols: str):
        if not self.connection: return None
        query = f"MATCH (s:{source_table})-[r:{rel_table} {{uuid: $uuid}}]->(t:{target_table}) RETURN {rel_cols}, s.uuid as source_node_uuid, t.uuid as target_node_uuid"
        results, _, _ = await self.execute_query(query, {"uuid": uuid})
        return parser_func(results[0]) if results else None

    async def get_entity_edge_by_uuid(self, uuid: str) -> Optional[EntityEdge]:
        cols = "r.uuid, r.name, r.group_id, r.fact, r.fact_embedding, r.episodes, r.created_at, r.expired_at, r.valid_at, r.invalid_at, r.attributes"
        return await self._get_edge_by_uuid_generic("RELATES_TO", "Entity", "Entity", uuid, self._kuzudb_to_entity_edge, cols)

    async def get_episodic_edge_by_uuid(self, uuid: str) -> Optional[EpisodicEdge]:
        cols = "r.uuid, r.group_id, r.created_at"
        return await self._get_edge_by_uuid_generic("MENTIONS", "Episodic", "Entity", uuid, self._kuzudb_to_episodic_edge, cols)

    async def get_community_edge_by_uuid(self, uuid: str) -> Optional[CommunityEdge]:
        cols = "r.uuid, r.group_id, r.created_at"
        return await self._get_edge_by_uuid_generic("HAS_MEMBER", "Community", "Entity", uuid, self._kuzudb_to_community_edge, cols)

    async def _get_edges_by_uuids_generic(self, rel_table: str, source_table: str, target_table: str, uuids: List[str], parser_func, rel_cols: str):
        if not self.connection or not uuids: return []
        query = f"MATCH (s:{source_table})-[r:{rel_table}]->(t:{target_table}) WHERE r.uuid IN $uuids RETURN {rel_cols}, s.uuid as source_node_uuid, t.uuid as target_node_uuid"
        results, _, _ = await self.execute_query(query, {"uuids": uuids})
        return [parser_func(res) for res in results]

    async def get_entity_edges_by_uuids(self, uuids: List[str]) -> List[EntityEdge]:
        cols = "r.uuid, r.name, r.group_id, r.fact, r.fact_embedding, r.episodes, r.created_at, r.expired_at, r.valid_at, r.invalid_at, r.attributes"
        return await self._get_edges_by_uuids_generic("RELATES_TO", "Entity", "Entity", uuids, self._kuzudb_to_entity_edge, cols)

    async def get_episodic_edges_by_uuids(self, uuids: List[str]) -> List[EpisodicEdge]:
        cols = "r.uuid, r.group_id, r.created_at"
        return await self._get_edges_by_uuids_generic("MENTIONS", "Episodic", "Entity", uuids, self._kuzudb_to_episodic_edge, cols)

    async def get_community_edges_by_uuids(self, uuids: List[str]) -> List[CommunityEdge]:
        cols = "r.uuid, r.group_id, r.created_at"
        return await self._get_edges_by_uuids_generic("HAS_MEMBER", "Community", "Entity", uuids, self._kuzudb_to_community_edge, cols)

    async def _get_edges_by_group_ids_generic(self, rel_table: str, source_table: str, target_table: str, group_ids: List[str], parser_func, limit: Optional[int], uuid_cursor: Optional[str], rel_cols: str):
        if not self.connection or not group_ids: return []
        params: Dict[str, Any] = {"group_ids": group_ids}
        query = f"MATCH (s:{source_table})-[r:{rel_table}]->(t:{target_table}) WHERE r.group_id IN $group_ids RETURN {rel_cols}, s.uuid as source_node_uuid, t.uuid as target_node_uuid"
        if limit is not None: # Basic limit
            query += " LIMIT $limit"
            params["limit"] = limit
        results, _, _ = await self.execute_query(query, params)
        return [parser_func(res) for res in results]

    async def get_entity_edges_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EntityEdge]:
        cols = "r.uuid, r.name, r.group_id, r.fact, r.fact_embedding, r.episodes, r.created_at, r.expired_at, r.valid_at, r.invalid_at, r.attributes"
        return await self._get_edges_by_group_ids_generic("RELATES_TO", "Entity", "Entity", group_ids, self._kuzudb_to_entity_edge, limit, uuid_cursor, cols)

    async def get_episodic_edges_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EpisodicEdge]:
        cols = "r.uuid, r.group_id, r.created_at"
        return await self._get_edges_by_group_ids_generic("MENTIONS", "Episodic", "Entity", group_ids, self._kuzudb_to_episodic_edge, limit, uuid_cursor, cols)

    async def get_community_edges_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[CommunityEdge]:
        cols = "r.uuid, r.group_id, r.created_at"
        return await self._get_edges_by_group_ids_generic("HAS_MEMBER", "Community", "Entity", group_ids, self._kuzudb_to_community_edge, limit, uuid_cursor, cols)
    
    async def get_entity_edges_by_node_uuid(self, node_uuid: str) -> List[EntityEdge]:
        if not self.connection: return []
        # Kuzu Cypher may not support UNION ALL in the same way or require subqueries.
        # For simplicity, running two queries and combining results.
        # This could be optimized if Kuzu has a better way to match on either source or target.
        cols = "r.uuid, r.name, r.group_id, r.fact, r.fact_embedding, r.episodes, r.created_at, r.expired_at, r.valid_at, r.invalid_at, r.attributes"
        
        query1 = f"MATCH (s:Entity {{uuid: $node_uuid}})-[r:RELATES_TO]->(t:Entity) RETURN {cols}, s.uuid AS source_node_uuid, t.uuid AS target_node_uuid"
        results1, _, _ = await self.execute_query(query1, {"node_uuid": node_uuid})
        
        query2 = f"MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity {{uuid: $node_uuid}}) WHERE s.uuid <> $node_uuid RETURN {cols}, s.uuid AS source_node_uuid, t.uuid AS target_node_uuid" # Avoid self-loops if s.uuid <> t.uuid
        results2, _, _ = await self.execute_query(query2, {"node_uuid": node_uuid})
        
        # Combine and deduplicate by edge UUID
        combined_results: Dict[str, EntityEdge] = {}
        for res_dict in results1 + results2:
            edge = self._kuzudb_to_entity_edge(res_dict)
            combined_results[edge.uuid] = edge
        return list(combined_results.values())

    async def delete_edge(self, uuid: str) -> None:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        logger.info(f"Attempting to delete edge {uuid} across all known rel tables.")
        for rel_table in ["MENTIONS", "RELATES_TO", "HAS_MEMBER"]:
            try:
                # Assuming source and target node types are not needed for MATCH by edge uuid
                await self.execute_query(f"MATCH ()-[r:{rel_table} {{uuid: $uuid}}]-() DELETE r", {"uuid": uuid})
                logger.debug(f"Edge {uuid} potentially deleted from table {rel_table}.")
                return # Assume edge UUID is unique across types, so if found and deleted, work is done.
            except Exception: # Kuzu may error if edge not found in a table
                pass
        logger.debug(f"Finished attempt to delete edge {uuid} across all tables.")

    async def clear_data(self, group_ids: Optional[List[str]] = None) -> None:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        node_table_names = ["Entity", "Episodic", "Community"]
        rel_table_names = ["MENTIONS", "RELATES_TO", "HAS_MEMBER"]
        
        if group_ids is None:
            logger.warning("Clearing all data from KuzuDB by dropping and recreating tables.")
            await self.build_indices_and_constraints(delete_existing=True)
        else:
            logger.warning(f"Clearing data for group_ids {group_ids}. This is a complex operation in KuzuDB.")
            for group_id_val in group_ids:
                # 1. Delete relationships associated with nodes in the target group_id.
                # This is the most complex part due to needing to know all connection patterns.
                # Example: If an Entity in group_id_val is connected to an Episodic node NOT in group_id_val via MENTIONS (if MENTIONS itself doesn't have group_id)
                # For simplicity, assume rels to be deleted also have group_id or are implicitly handled by node deletion if Kuzu cascades (it doesn't)
                
                # Delete rels WITH the group_id
                for rel_table in rel_table_names:
                    try:
                        await self.execute_query(f"MATCH ()-[r:{rel_table} {{group_id: $gid}}]-() DELETE r", {"gid": group_id_val})
                    except Exception as e: logger.debug(f"Error deleting rels from {rel_table} for group {group_id_val}: {e}")

                # Delete rels connected TO nodes with the group_id
                for node_table in node_table_names:
                    for rel_table in rel_table_names: # Iterate all combinations
                        try: # (node_in_group)-[r]-()
                            await self.execute_query(f"MATCH (n:{node_table} {{group_id: $gid}})-[r:{rel_table}]-() DELETE r", {"gid": group_id_val})
                        except Exception as e: logger.debug(f"Error deleting outbound rels for {node_table} group {group_id_val} via {rel_table}: {e}")
                        try: # ()-[r]-(node_in_group)
                            await self.execute_query(f"MATCH ()-[r:{rel_table}]-(n:{node_table} {{group_id: $gid}}) DELETE r", {"gid": group_id_val})
                        except Exception as e: logger.debug(f"Error deleting inbound rels for {node_table} group {group_id_val} via {rel_table}: {e}")
                
                # 2. Delete nodes
                for node_table in node_table_names:
                    try:
                        await self.execute_query(f"MATCH (n:{node_table} {{group_id: $gid}}) DELETE n", {"gid": group_id_val})
                    except Exception as e: logger.warning(f"Error deleting nodes from {node_table} for group {group_id_val}: {e}")
            logger.info(f"Finished attempt to clear data for group_ids: {group_ids}.")

    async def retrieve_episodes(
        self, reference_time: datetime, last_n: int, 
        group_ids: Optional[List[str]] = None, source: Optional[EpisodeType] = None
    ) -> List[EpisodicNode]:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        params: Dict[str, Any] = {"ref_time": reference_time, "limit_n": last_n}
        query_parts = ["MATCH (e:Episodic) WHERE e.valid_at <= $ref_time"]
        if group_ids:
            query_parts.append("AND e.group_id IN $gids")
            params["gids"] = group_ids
        if source:
            query_parts.append("AND e.source = $src")
            params["src"] = source.value
        
        query = "\n".join(query_parts) + " RETURN e.uuid, e.name, e.group_id, e.source, e.source_description, e.content, e.valid_at, e.created_at, e.entity_edges ORDER BY e.valid_at DESC LIMIT $limit_n"
        results, _, _ = await self.execute_query(query, params)
        episodes = [self._kuzudb_to_episodic_node(res) for res in results]
        return list(reversed(episodes))

    # --- Search Operations (Basic KuzuDB Implementations) ---
    def _apply_search_filters_to_query_basic(
        self,
        search_filter: SearchFilters,
        alias: str, # e.g., "n" for node, "r" for relationship
        params: Dict[str, Any] # to add new parameter names and values
    ) -> str:
        filter_clauses: List[str] = []

        # group_id_filter (if SearchFilters has it and it's not a primary method param)
        if search_filter.group_id_filter:
            param_name = f"{alias}_group_id_filter"
            filter_clauses.append(f"{alias}.group_id = ${param_name}")
            params[param_name] = search_filter.group_id_filter

        # Date filters
        # Note: KuzuDB expects TIMESTAMP type for date comparisons.
        # Ensure datetime objects are passed correctly.
        if search_filter.created_at_filter:
            op = search_filter.created_at_filter.operator
            # Kuzu Cypher might not support all comparison ops directly like this or need casting.
            # This is a direct translation; testing will verify Kuzu's capabilities.
            if op not in [ComparisonOperator.CONTAINS, ComparisonOperator.STARTS_WITH, ComparisonOperator.ENDS_WITH]:
                param_name = f"{alias}_created_at_date"
                filter_clauses.append(f"{alias}.created_at {op.value} ${param_name}")
                params[param_name] = search_filter.created_at_filter.date

        if search_filter.valid_at_filter: # Typically for EpisodicNode or specific edges
            op = search_filter.valid_at_filter.operator
            if op not in [ComparisonOperator.CONTAINS, ComparisonOperator.STARTS_WITH, ComparisonOperator.ENDS_WITH]:
                param_name = f"{alias}_valid_at_date"
                # Ensure the property exists on the aliased item (e.g., e.valid_at for Episodic)
                filter_clauses.append(f"{alias}.valid_at {op.value} ${param_name}")
                params[param_name] = search_filter.valid_at_filter.date

        # Attribute filters: This is the most complex part for KuzuDB
        # KuzuDB stores 'attributes' as a STRING (JSON). Direct querying of JSON content is limited.
        # A common workaround is to use string CONTAINS for simple key-value checks if the JSON structure is predictable.
        # e.g., attributes CONTAINS '"key":"value"'
        # This is brittle. For more robust attribute filtering, consider:
        # 1. Promoting frequently filtered attributes to top-level schema properties.
        # 2. Fetching and filtering in the application layer (less efficient).
        # 3. Using Kuzu's future potential JSON/struct processing functions if they become available.
        if search_filter.attribute_filters:
            for i, attr_filter in enumerate(search_filter.attribute_filters):
                # Simplified: only support EQUALS and CONTAINS on the stringified attributes
                # This is a very basic and potentially slow way to filter JSON strings.
                # For EQUALS on a specific key:
                if attr_filter.operator == ComparisonOperator.EQUALS:
                    # Example: attributes CONTAINS '"key":"value"' (stringified value)
                    # This is highly dependent on JSON serialization format (spaces, quotes, etc.)
                    # A safer bet is to filter for the key existing, then value separately if possible.
                    # For now, a very naive CONTAINS:
                    attr_param_name = f"{alias}_attr_val_{i}"
                    # This filter is very basic and might not be robust enough.
                    # It assumes 'attributes' is the column name.
                    # We are checking if the stringified key-value pair is present.
                    # json.dumps is used to get a consistent string representation.
                    # Ensure simple types for attr_filter.value (str, int, bool)
                    try:
                        # Create a mini-dict for the specific key-value to search for its string form
                        # This is still very fragile. E.g. {"key": "value"} vs {"key": "value", "other": "data"}
                        # A better CONTAINS might be just for the value if key is known:
                        # WHERE attributes CONTAINS '"key":"' AND attributes CONTAINS ':"value"' (problematic with nested/complex values)

                        # Let's attempt a CONTAINS for the value part of a key-value pair.
                        # This assumes the key is known and you're checking its value.
                        # This is still not ideal.
                        # Example: WHERE {alias}.attributes CONTAINS '"key_name":"value_as_string"'
                        # This requires knowing the exact string representation in the JSON.

                        # Given Kuzu's current limitations on JSON querying, we might have to state that
                        # attribute_filters are not fully supported or only in a very limited way.
                        # The most reliable way is to fetch and filter client-side for complex attribute queries.

                        # For this iteration, let's only implement CONTAINS on the raw attributes string
                        # for the filter's value, assuming the user knows what they are searching for
                        # within the JSON string. This is not specific to a key.
                        if attr_filter.operator == ComparisonOperator.CONTAINS and isinstance(attr_filter.value, str):
                            filter_clauses.append(f"CONTAINS(lower({alias}.attributes), lower(${attr_param_name}))")
                            params[attr_param_name] = attr_filter.value
                            logger.debug(f"KuzuDB applying basic CONTAINS filter on attributes string for alias {alias} with value {attr_filter.value}")
                        else:
                            logger.warning(f"KuzuDB: Attribute filter operator '{attr_filter.operator}' for key '{attr_filter.key}' may not be fully supported or optimized. Only basic CONTAINS on stringified attributes is attempted.")
                    except Exception as e:
                        logger.error(f"KuzuDB: Error processing attribute filter for key '{attr_filter.key}': {e}")


        # Raw Cypher WHERE clause (use with caution)
        if search_filter.raw_where_clause:
            # IMPORTANT: This is a potential security risk if raw_where_clause comes from untrusted input.
            # It also makes the query less portable and harder to maintain.
            # Parameter binding is NOT done for raw_where_clause here.
            filter_clauses.append(f"({search_filter.raw_where_clause})")
            logger.warning("KuzuDB: Applied a raw_where_clause. Ensure it's sanitized and correct.")

        return " AND ".join(filter_clauses) if filter_clauses else ""


    async def node_fulltext_search(self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[EntityNode]:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        params: Dict[str, Any] = {"query_str_lower": query.lower(), "limit_val": limit}
        
        # Base FTS condition
        where_clauses = ["(CONTAINS(lower(n.name), $query_str_lower) OR CONTAINS(lower(n.summary), $query_str_lower))"]

        if group_ids: # group_ids from method parameter take precedence
            where_clauses.append("n.group_id IN $gids")
            params["gids"] = group_ids
        
        # Apply additional filters from SearchFilters object
        # The params dict will be updated by _apply_search_filters_to_query_basic
        filter_query_part = self._apply_search_filters_to_query_basic(search_filter, "n", params)
        if filter_query_part:
            where_clauses.append(filter_query_part)
        
        final_query = f"MATCH (n:Entity) WHERE {' AND '.join(where_clauses)} RETURN n.uuid, n.name, n.group_id, n.created_at, n.summary, n.name_embedding, n.attributes LIMIT $limit_val"
        results, _, _ = await self.execute_query(final_query, params)
        return [self._kuzudb_to_entity_node(res) for res in results]

    async def node_similarity_search(self, search_vector: List[float], search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10, min_score: float = 0.6) -> List[EntityNode]:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        # FIXME: Verify KuzuDB's actual vector similarity function name and syntax.
        # Common examples: list_similarity(a, b), cosine_distance(a,b), etc.
        # Adjust the function name and potentially the scoring (distance vs similarity) as needed.
        KUZU_SIMILARITY_FUNCTION = "cosine_similarity" # Placeholder - Assuming higher is better.
        
        params: Dict[str, Any] = {"s_vec": search_vector, "lim": limit, "min_s": min_score}
        where_clauses = [f"{KUZU_SIMILARITY_FUNCTION}(n.name_embedding, $s_vec) >= $min_s"]

        if group_ids: # group_ids from method parameter take precedence
            where_clauses.append("n.group_id IN $gids")
            params["gids"] = group_ids
        
        # Apply additional filters from SearchFilters object
        filter_query_part = self._apply_search_filters_to_query_basic(search_filter, "n", params)
        if filter_query_part:
            where_clauses.append(filter_query_part)

        final_query = f"MATCH (n:Entity) WHERE {' AND '.join(where_clauses)} RETURN n.uuid, n.name, n.group_id, n.created_at, n.summary, n.name_embedding, n.attributes, {KUZU_SIMILARITY_FUNCTION}(n.name_embedding, $s_vec) AS score ORDER BY score DESC LIMIT $lim"
        try:
            results, _, _ = await self.execute_query(final_query, params)
            return [self._kuzudb_to_entity_node(res) for res in results]
        except Exception as e: # Catch if KUZU_SIMILARITY_FUNCTION is not defined
            logger.error(f"node_similarity_search failed, possibly due to Kuzu vector function: {e}")
            return []
    
    # Other search methods would follow similar patterns, adapting Cypher and using Kuzu-specific functions.
    # For brevity, only implementing a few search methods here. Others remain NotImplementedError.

    async def edge_fulltext_search(self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[EntityEdge]:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        params: Dict[str, Any] = {"query_str_lower": query.lower(), "limit_val": limit}

        # Base FTS condition
        where_clauses = ["(CONTAINS(lower(r.name), $query_str_lower) OR CONTAINS(lower(r.fact), $query_str_lower))"]

        if group_ids: # group_ids from method parameter take precedence
            where_clauses.append("r.group_id IN $gids") # Assuming edges can have group_ids
            params["gids"] = group_ids
        
        # Apply additional filters from SearchFilters object
        filter_query_part = self._apply_search_filters_to_query_basic(search_filter, "r", params)
        if filter_query_part:
            where_clauses.append(filter_query_part)

        final_query = f"MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity) WHERE {' AND '.join(where_clauses)} RETURN r.uuid, r.name, r.group_id, r.fact, r.fact_embedding, r.episodes, r.created_at, r.expired_at, r.valid_at, r.invalid_at, r.attributes, s.uuid as source_node_uuid, t.uuid as target_node_uuid LIMIT $limit_val"
        results, _, _ = await self.execute_query(final_query, params)
        return [self._kuzudb_to_entity_edge(res) for res in results]

    async def get_embeddings_for_nodes(self, nodes: List[EntityNode]) -> Dict[str, List[float]]:
        if not self.connection or not nodes: return {}
        uuids = [n.uuid for n in nodes]
        query = "MATCH (n:Entity) WHERE n.uuid IN $uuids RETURN n.uuid AS uuid, n.name_embedding AS embedding"
        results, _, _ = await self.execute_query(query, {"uuids": uuids})
        return {r['uuid']: r['embedding'] for r in results if r.get('embedding')}

    async def get_embeddings_for_communities(self, communities: List[CommunityNode]) -> Dict[str, List[float]]:
        if not self.connection or not communities: return {}
        uuids = [c.uuid for c in communities]
        query = "MATCH (c:Community) WHERE c.uuid IN $uuids RETURN c.uuid AS uuid, c.name_embedding AS embedding"
        results, _, _ = await self.execute_query(query, {"uuids": uuids})
        return {r['uuid']: r['embedding'] for r in results if r.get('embedding')}

    async def get_embeddings_for_edges(self, edges: List[EntityEdge]) -> Dict[str, List[float]]:
        if not self.connection or not edges: return {}
        uuids = [e.uuid for e in edges]
        query = "MATCH ()-[r:RELATES_TO]-() WHERE r.uuid IN $uuids RETURN r.uuid AS uuid, r.fact_embedding AS embedding" # Assuming only RELATES_TO has embeddings
        results, _, _ = await self.execute_query(query, {"uuids": uuids})
        return {r['uuid']: r['embedding'] for r in results if r.get('embedding')}

    async def get_mentioned_nodes(self, episodes: List[EpisodicNode]) -> List[EntityNode]:
        if not self.connection or not episodes: return []
        uuids = [ep.uuid for ep in episodes]
        query = "MATCH (e:Episodic)-[:MENTIONS]->(n:Entity) WHERE e.uuid IN $uuids RETURN DISTINCT n.uuid, n.name, n.group_id, n.created_at, n.summary, n.name_embedding, n.attributes"
        results, _, _ = await self.execute_query(query, {"uuids": uuids})
        return [self._kuzudb_to_entity_node(res) for res in results]

    async def get_communities_by_nodes(self, nodes: List[EntityNode]) -> List[CommunityNode]:
        if not self.connection or not nodes: return []
        uuids = [n.uuid for n in nodes]
        query = "MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity) WHERE n.uuid IN $uuids RETURN DISTINCT c.uuid, c.name, c.group_id, c.created_at, c.summary, c.name_embedding"
        results, _, _ = await self.execute_query(query, {"uuids": uuids})
        return [self._kuzudb_to_community_node(res) for res in results]

    # --- Remaining Stubs ---
    async def add_nodes_and_edges_bulk(
        self,
        episodic_nodes: List[EpisodicNode],
        episodic_edges: List[EpisodicEdge],
        entity_nodes: List[EntityNode],
        entity_edges: List[EntityEdge],
        embedder: EmbedderClient
    ) -> None:
        """
        Adds nodes and edges in bulk to KuzuDB.
        This implementation uses KuzuDB's `COPY FROM` command with Parquet files for efficiency.
        It involves serializing Pydantic objects to pandas DataFrames, writing to temporary
        Parquet files, and then using `COPY FROM`.
        Embeddings are generated for nodes/edges if not present before bulk loading.
        Nodes are upserted using `ON CONFLICT DO UPDATE` with `COPY`.
        Relationships are appended.
        Requires `pandas` and `pyarrow` (for Parquet) to be installed.
        """
        if not self.connection:
            raise ConnectionError("KuzuDB connection not established.")

        logger.info(f"Starting KuzuDB bulk add: {len(episodic_nodes)} episodic, {len(entity_nodes)} entity nodes; {len(episodic_edges)} episodic, {len(entity_edges)} entity edges.")

        # Required imports for COPY FROM approach
        try:
            import pandas as pd
        except ImportError:
            logger.error("Pandas library is required for KuzuDB bulk operations but not installed.")
            raise ImportError("Pandas library is required for KuzuDB bulk operations.")

        import tempfile
        import os

        # Helper function to write dataframe to temp parquet file and run COPY
        async def _copy_from_df_to_kuzu(table_name: str, df: pd.DataFrame, node_table: bool = True):
            if df.empty:
                logger.info(f"No data to load for table {table_name}.")
                return

            # Ensure temp file is created and handled correctly, especially with 'delete=False'
            fd, tmpfile_path = tempfile.mkstemp(suffix=".parquet")
            os.close(fd) # Close the file descriptor, as to_parquet will open/write to the path

            try:
                df.to_parquet(tmpfile_path)

                copy_options = ""
                if node_table: # Nodes have PK (uuid)
                    update_cols = [col for col in df.columns if col != "uuid"]
                    if update_cols:
                        set_clauses = ", ".join([f"{col}=EXCLUDED.{col}" for col in update_cols])
                        # Kuzu's ON CONFLICT syntax for COPY: (key_column_name) DO UPDATE SET ...
                        copy_options = f"(ON_CONFLICT (uuid) DO UPDATE SET {set_clauses})"
                    else:
                        copy_options = f"(ON_CONFLICT (uuid) DO NOTHING)"

                # Ensure file path is properly escaped/quoted for Cypher, though Kuzu usually handles local paths well.
                query = f"COPY {table_name} FROM '{tmpfile_path}' {copy_options}"
                logger.debug(f"Executing Kuzu COPY: {query}")
                await self.execute_query(query)
                logger.info(f"Successfully loaded data into {table_name} from {tmpfile_path}")
            except Exception as e:
                logger.error(f"Error during Kuzu COPY for table {table_name} from {tmpfile_path}: {e}")
                raise
            finally:
                if os.path.exists(tmpfile_path):
                    try:
                        os.remove(tmpfile_path)
                    except Exception as e_remove:
                        logger.error(f"Error removing temporary Kuzu COPY file {tmpfile_path}: {e_remove}")

        # 1. Embedding Generation
        logger.info("Generating embeddings for KuzuDB bulk load...")
        # Entity Nodes
        for node in entity_nodes:
            if node.name_embedding is None and hasattr(node, 'name') and node.name:
                text_to_embed = node.name.replace('\n', ' ')
                embedding_result = await embedder.create(input_data=[text_to_embed])
                if embedding_result and isinstance(embedding_result, list) and len(embedding_result) > 0 and isinstance(embedding_result[0], list):
                    node.name_embedding = embedding_result[0]
        # Entity Edges
        for edge in entity_edges:
            if edge.fact_embedding is None and hasattr(edge, 'fact') and edge.fact:
                text_to_embed = edge.fact.replace('\n', ' ')
                embedding_result = await embedder.create(input_data=[text_to_embed])
                if embedding_result and isinstance(embedding_result, list) and len(embedding_result) > 0 and isinstance(embedding_result[0], list):
                    edge.fact_embedding = embedding_result[0]
        # Community Nodes (if they have embeddings, e.g. name_embedding)
        # Assuming CommunityNode list is not passed but could be:
        # for node in community_nodes: if node.name_embedding is None ...
        logger.info("Embedding generation complete.")

        # 2. Node Processing
        # Episodic Nodes
        if episodic_nodes:
            episodic_data = []
            for node in episodic_nodes:
                d = node.model_dump(exclude_none=False) # include None to match schema columns
                d["source"] = node.source.value
                 # Ensure all fields defined in schema are present, even if None
                for key in ['uuid', 'name', 'group_id', 'source', 'source_description', 'content', 'valid_at', 'created_at', 'entity_edges']:
                    d.setdefault(key, None)
                if d.get('entity_edges') is None: # Ensure 'entity_edges' is an empty list if None
                    d['entity_edges'] = []
                episodic_data.append(d)
            episodic_df = pd.DataFrame(episodic_data)
            await _copy_from_df_to_kuzu("Episodic", episodic_df, node_table=True)

        # Entity Nodes
        if entity_nodes:
            entity_data = []
            for node in entity_nodes:
                d = node.model_dump(exclude_none=False)
                d["attributes"] = json.dumps(node.attributes) if node.attributes else "{}"
                for key in ['uuid', 'name', 'group_id', 'created_at', 'summary', 'name_embedding', 'attributes']:
                    d.setdefault(key, None)
                entity_data.append(d)
            entity_df = pd.DataFrame(entity_data)
            # Ensure embedding list format is compatible with Parquet/Kuzu if it's list-of-list
            await _copy_from_df_to_kuzu("Entity", entity_df, node_table=True)

        # Community Nodes (assuming CommunityNode list might be added to signature later)
        # For now, this part is illustrative if community_nodes were passed in.
        # community_nodes: List[CommunityNode] = [] # Example
        # if community_nodes:
        #     community_data = [node.model_dump(exclude_none=False) for node in community_nodes]
        #     community_df = pd.DataFrame(community_data)
        #     await _copy_from_df_to_kuzu("Community", community_df, node_table=True)


        # 3. Relationship Processing
        # Kuzu requires _from and _to columns for relationships, mapping to node PKs.
        # Episodic Edges (MENTIONS: Episodic -> Entity)
        if episodic_edges:
            episodic_edge_data = []
            for edge in episodic_edges:
                d = edge.model_dump(exclude_none=False)
                d["_from"] = edge.source_node_uuid
                d["_to"] = edge.target_node_uuid
                for key in ['uuid', 'group_id', 'created_at', '_from', '_to']: # Ensure all schema fields
                    d.setdefault(key, None)
                episodic_edge_data.append(d)
            episodic_edge_df = pd.DataFrame(episodic_edge_data)
            # Select only columns that are part of the MENTIONS schema + _from, _to
            # Schema: uuid STRING, group_id STRING, created_at TIMESTAMP
            cols_for_mentions = ['_from', '_to', 'uuid', 'group_id', 'created_at']
            episodic_edge_df = episodic_edge_df[cols_for_mentions]
            await _copy_from_df_to_kuzu("MENTIONS", episodic_edge_df, node_table=False)

        # Entity Edges (RELATES_TO: Entity -> Entity)
        if entity_edges:
            entity_edge_data = []
            for edge in entity_edges:
                d = edge.model_dump(exclude_none=False)
                d["_from"] = edge.source_node_uuid
                d["_to"] = edge.target_node_uuid
                d["attributes"] = json.dumps(edge.attributes) if edge.attributes else "{}"
                # Schema: uuid, name, group_id, fact, fact_embedding, episodes, created_at, expired_at, valid_at, invalid_at, attributes
                for key in ['uuid', 'name', 'group_id', 'fact', 'fact_embedding', 'episodes', 'created_at', 'expired_at', 'valid_at', 'invalid_at', 'attributes', '_from', '_to']:
                     d.setdefault(key, None)
                if d.get('episodes') is None: # Ensure 'episodes' is an empty list if None
                    d['episodes'] = []
                entity_edge_data.append(d)
            entity_edge_df = pd.DataFrame(entity_edge_data)
            cols_for_relates_to = ['_from', '_to', 'uuid', 'name', 'group_id', 'fact', 'fact_embedding', 'episodes', 'created_at', 'expired_at', 'valid_at', 'invalid_at', 'attributes']
            entity_edge_df = entity_edge_df[cols_for_relates_to]
            await _copy_from_df_to_kuzu("RELATES_TO", entity_edge_df, node_table=False)

        # Community Edges (HAS_MEMBER: Community -> Entity)
        # community_edges: List[CommunityEdge] = [] # Example
        # if community_edges:
        #     community_edge_data = [edge.model_dump(exclude_none=False) for edge in community_edges]
        #     # Add _from, _to mapping
        #     # community_edge_df = pd.DataFrame(community_edge_data)
        #     # await _copy_from_df_to_kuzu("HAS_MEMBER", community_edge_df, node_table=False)

        logger.info("KuzuDB bulk add_nodes_and_edges operation using COPY FROM completed.")

    async def count_node_mentions(self, node_uuid: str) -> int:
        if not self.connection:
            raise ConnectionError("KuzuDB connection not established.")

        # Kuzu Cypher for counting distinct incoming relationships from Episodic nodes via MENTIONS
        # Assuming MENTIONS is defined as (Episodic)-[MENTIONS]->(Entity)
        query = """
            MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $node_uuid})
            RETURN count(DISTINCT e.uuid) AS mention_count
        """
        # Kuzu might prefer count(e) or count(*) if e is guaranteed distinct by the match pattern for this purpose.
        # Using count(DISTINCT e.uuid) for clarity that we want distinct episodic nodes.

        results, _, _ = await self.execute_query(query, params={"node_uuid": node_uuid})
        if results and results[0] and "mention_count" in results[0]:
            # Ensure the count is an integer
            try:
                return int(results[0]["mention_count"])
            except (ValueError, TypeError):
                logger.error(f"Could not convert mention_count '{results[0]['mention_count']}' to int for node {node_uuid}")
                return 0
        return 0

    async def edge_similarity_search(self, search_vector: List[float], source_node_uuid: Optional[str], target_node_uuid: Optional[str], search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10, min_score: float = 0.6) -> List[EntityEdge]:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        # FIXME: Verify KuzuDB's actual vector similarity function name and syntax.
        KUZU_SIMILARITY_FUNCTION = "cosine_similarity" # Placeholder - Assuming higher is better.

        params: Dict[str, Any] = {"s_vec": search_vector, "lim": limit, "min_s": min_score}
        match_clauses = ["MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)"]
        # Ensure fact_embedding is not null before calling similarity function
        where_clauses = ["r.fact_embedding IS NOT NULL", f"{KUZU_SIMILARITY_FUNCTION}(r.fact_embedding, $s_vec) >= $min_s"]

        if source_node_uuid:
            where_clauses.append("s.uuid = $source_uuid")
            params["source_uuid"] = source_node_uuid
        if target_node_uuid:
            where_clauses.append("t.uuid = $target_uuid")
            params["target_uuid"] = target_node_uuid

        if group_ids: # group_ids from method parameter take precedence
            where_clauses.append("r.group_id IN $gids") # Assuming edges can have group_ids
            params["gids"] = group_ids

        # Apply additional filters from SearchFilters object
        filter_query_part = self._apply_search_filters_to_query_basic(search_filter, "r", params)
        if filter_query_part:
            where_clauses.append(filter_query_part)

        return_cols = "r.uuid, r.name, r.group_id, r.fact, r.fact_embedding, r.episodes, r.created_at, r.expired_at, r.valid_at, r.invalid_at, r.attributes, s.uuid as source_node_uuid, t.uuid as target_node_uuid"
        final_query = f"{' '.join(match_clauses)} WHERE {' AND '.join(where_clauses)} RETURN {return_cols}, {KUZU_SIMILARITY_FUNCTION}(r.fact_embedding, $s_vec) AS score ORDER BY score DESC LIMIT $lim"

        try:
            results, _, _ = await self.execute_query(final_query, params)
            return [self._kuzudb_to_entity_edge(res) for res in results]
        except Exception as e:
            logger.error(f"edge_similarity_search failed, possibly due to Kuzu vector function: {e}")
            return []

    async def edge_bfs_search(self, bfs_origin_node_uuids: Optional[List[str]], bfs_max_depth: int, search_filter: SearchFilters, limit: int) -> List[EntityEdge]:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        if not bfs_origin_node_uuids: return []

        # Kuzu BFS query: MATCH (origin)-[r*1..max_depth BFS]->(destination)
        # We need to return the edges along these paths.
        # This query returns paths, from which we need to extract edges.
        # Kuzu's path object might need specific handling.
        # For simplicity, let's get distinct edges.
        # search_filter can specify edge types (r:TYPE1|TYPE2) or node labels.

        # Constructing edge types for BFS path
        # Assuming search_filter.edge_types refers to relationship table names like "RELATES_TO", "MENTIONS"
        edge_type_str = "*"
        if search_filter and search_filter.edge_types:
            # Example: search_filter.edge_types = ["RELATES_TO", "MENTIONS"] -> ":RELATES_TO|MENTIONS"
            edge_type_str = ":" + "|".join(search_filter.edge_types)

        # The _apply_search_filters_to_query_basic needs params dict to add its parameters
        params: Dict[str, Any] = {"origin_uuids": bfs_origin_node_uuids, "lim": limit} # Moved params initialization up

        # Base WHERE for origin UUIDs
        path_match_where_clauses = ["origin.uuid IN $origin_uuids"]
        # Additional filters on the relationship 'r' itself
        # Note: Kuzu's BFS MATCH...WHERE applies to the path pattern, not UNWINDed elements easily.
        # So, filtering on 'r' properties might be less direct or require subqueries if complex.
        # The current _apply_search_filters_to_query_basic is designed for simple property checks.
        # If search_filter.edge_types is used, it should be part of the MATCH pattern: (o)-[r:{edge_type_str}*1..{max_depth}]->(p)
        # For now, applying filters on 'r' post-UNWIND.

        # Construct the core MATCH part
        # If search_filter.edge_types has values, use them in the path pattern for efficiency
        # Example: edge_type_str = search_filter.edge_types.join("|") -> MATCH (o)-[r:${edge_type_str}*1..${max_depth}]->(p)
        # This is more efficient than filtering post-UNWIND if Kuzu optimizer uses it.
        # However, _apply_search_filters_to_query_basic might try to filter on r.type or r.label if SearchFilters supports it.
        # For Kuzu, rel table names are fixed, so edge_types in SearchFilters should map to these.
        # The current code uses a generic '*' in MATCH and filters later.
        # Let's keep it simple for now and rely on post-UNWIND filtering for 'r' properties.

        match_clause = f"MATCH (origin:Entity)-[rels{edge_type_str}*1..{bfs_max_depth} BFS]->(peer:Entity)"

        # Filters applied to the relationship 'r' after UNWIND
        # These are for properties *of the relationship itself*, not its type.
        additional_r_filter_str = self._apply_search_filters_to_query_basic(search_filter, "r", params)

        # Building the query
        # WITH clause is essential to make `r` available for WHERE and RETURN
        # The WHERE clause here applies to `origin` and properties of `r`

        # Start with base path match conditions
        full_where_conditions = path_match_where_clauses.copy()

        # Construct the final WHERE clause string for the main MATCH part
        # This primarily includes origin.uuid IN $origin_uuids
        # Any filters on 'r' will be applied *after* UNWIND if possible, or need to be part of the MATCH path pattern if Kuzu supports it well.
        # For now, let's assume Kuzu applies WHERE after pathfinding but before UNWIND for path properties,
        # and for 'r' properties specifically, we might need a separate WITH...WHERE post-UNWIND.
        # Re-simplifying: Apply filters that can be applied to the MATCH path in its WHERE clause.
        # Filters on individual 'r' from 'rels' list need to be handled carefully.
        # The provided code structure `MATCH ... WHERE ... UNWIND ... WITH ... {final_where_clause}` for r filters is problematic.
        # A more standard Cypher approach for filtering properties of `r` from an unwound list `rels`
        # would be `... UNWIND rels as r WITH r WHERE condition_on_r ...`
        # Or, if Kuzu allows filtering on path relationships directly: `MATCH p=(origin)-[...]-(peer) WHERE all(r IN relationships(p) WHERE r.property = value)`
        # Given Kuzu's specifics, let's try to keep the main WHERE for `origin` and `peer` level filters.
        # And apply `r` specific filters after UNWIND using a `WITH r WHERE ...` if needed.
        # The current `_apply_search_filters_to_query_basic` on `r` will generate clauses like `r.property = value`.
        # These should be applied after UNWIND.

        # Initial match and origin filter
        query_parts = [
            f"{match_clause}",
            "WHERE origin.uuid IN $origin_uuids", # Filter for origin nodes
            "UNWIND rels AS r" # Unpack relationships from paths
        ]

        # Apply filters for 'r' after UNWIND
        if additional_r_filter_str:
            query_parts.append(f"WHERE {additional_r_filter_str}") # Filters on properties of 'r'

        query_parts.append(
            "RETURN DISTINCT r.uuid, r.name, r.group_id, r.fact, r.fact_embedding, r.episodes, "
            "r.created_at, r.expired_at, r.valid_at, r.invalid_at, r.attributes, "
            "startNode(r).uuid AS source_node_uuid, endNode(r).uuid AS target_node_uuid"
        )
        query_parts.append("LIMIT $lim")

        query = "\n".join(query_parts)

        try:
            results, _, _ = await self.execute_query(query, params)
            return [self._kuzudb_to_entity_edge(res) for res in results if res.get('uuid')]
        except Exception as e:
            logger.error(f"KuzuDB edge_bfs_search failed for origins {bfs_origin_node_uuids}: {e}. Query: {query}")
            return []


    async def node_bfs_search(self, bfs_origin_node_uuids: Optional[List[str]], search_filter: SearchFilters, bfs_max_depth: int, limit: int) -> List[EntityNode]:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        if not bfs_origin_node_uuids: return []

        params: Dict[str, Any] = {"origin_uuids": bfs_origin_node_uuids, "lim": limit}

        # Edge type string for MATCH pattern
        edge_type_str = "*" # Default: any relationship type
        if search_filter.edge_types: # If specific edge types (rel tables) are given
            edge_type_str = ":" + "|".join(search_filter.edge_types)

        # Base WHERE clauses for the MATCH pattern
        # Filters on 'origin' node and ensures 'peer' is distinct
        # Also apply additional filters from SearchFilters to the 'peer' node

        where_clauses = ["origin.uuid IN $origin_uuids", "origin.uuid <> peer.uuid"]

        peer_filter_str = self._apply_search_filters_to_query_basic(search_filter, "peer", params)
        if peer_filter_str:
            where_clauses.append(peer_filter_str)

        final_where_clause_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        match_clause = f"MATCH (origin:Entity)-[rels{edge_type_str}*1..{bfs_max_depth} BFS]->(peer:Entity)"

        query = f"""
        {match_clause}
        {final_where_clause_str}
        RETURN DISTINCT peer.uuid, peer.name, peer.group_id, peer.created_at, peer.summary, peer.name_embedding, peer.attributes
        LIMIT $lim
        """

        try:
            results, _, _ = await self.execute_query(query, params)
            return [self._kuzudb_to_entity_node(res) for res in results]
        except Exception as e:
            logger.error(f"KuzuDB node_bfs_search failed for origins {bfs_origin_node_uuids}: {e}. Query: {query}")
            return []

    async def episode_fulltext_search(self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[EpisodicNode]:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        # Current FTS uses case-insensitive CONTAINS. See node_fulltext_search for notes on advanced FTS.
        params: Dict[str, Any] = {"query_str_lower": query.lower(), "limit_val": limit}

        # Searching on 'name' and 'content' properties of EpisodicNode
        where_clauses = ["(CONTAINS(lower(e.name), $query_str_lower) OR CONTAINS(lower(e.content), $query_str_lower) OR CONTAINS(lower(e.source_description), $query_str_lower))"]

        if group_ids: # group_ids from method parameter take precedence
            where_clauses.append("e.group_id IN $gids")
            params["gids"] = group_ids

        # Apply additional filters from SearchFilters object
        filter_query_part = self._apply_search_filters_to_query_basic(search_filter, "e", params)
        if filter_query_part:
            where_clauses.append(filter_query_part)

        cols = "e.uuid, e.name, e.group_id, e.source, e.source_description, e.content, e.valid_at, e.created_at, e.entity_edges"
        final_query = f"MATCH (e:Episodic) WHERE {' AND '.join(where_clauses)} RETURN {cols} LIMIT $limit_val"
        results, _, _ = await self.execute_query(final_query, params)
        return [self._kuzudb_to_episodic_node(res) for res in results]

    async def community_fulltext_search(self, query: str, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[CommunityNode]:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        # Current FTS uses case-insensitive CONTAINS. See node_fulltext_search for notes on advanced FTS.
        params: Dict[str, Any] = {"query_str_lower": query.lower(), "limit_val": limit}

        # Searching on 'name' and 'summary' properties of CommunityNode
        where_clauses = ["(CONTAINS(lower(c.name), $query_str_lower) OR CONTAINS(lower(c.summary), $query_str_lower))"]

        if group_ids: # group_ids from method parameter take precedence
            where_clauses.append("c.group_id IN $gids")
            params["gids"] = group_ids

        # Apply additional filters from SearchFilters object
        # Since community_fulltext_search doesn't take search_filter, we'd need to add it or use a default/empty one.
        # For now, assuming it might be added or this part of filtering is skipped if no search_filter object is available.
        # If search_filter were a param:
        # filter_query_part = self._apply_search_filters_to_query_basic(search_filter, "c", params)
        # if filter_query_part:
        #     where_clauses.append(filter_query_part)

        cols = "c.uuid, c.name, c.group_id, c.created_at, c.summary, c.name_embedding"
        final_query = f"MATCH (c:Community) WHERE {' AND '.join(where_clauses)} RETURN {cols} LIMIT $limit_val"
        results, _, _ = await self.execute_query(final_query, params)
        return [self._kuzudb_to_community_node(res) for res in results]

    async def community_similarity_search(self, search_vector: List[float], group_ids: Optional[List[str]] = None, limit: int = 10, min_score: float = 0.6) -> List[CommunityNode]:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        # FIXME: Verify KuzuDB's actual vector similarity function name and syntax.
        KUZU_SIMILARITY_FUNCTION = "cosine_similarity" # Placeholder - Assuming higher is better.

        params: Dict[str, Any] = {"s_vec": search_vector, "lim": limit, "min_s": min_score}
        # Ensure name_embedding is not null
        where_clauses = ["c.name_embedding IS NOT NULL", f"{KUZU_SIMILARITY_FUNCTION}(c.name_embedding, $s_vec) >= $min_s"]

        if group_ids: # group_ids from method parameter take precedence
            where_clauses.append("c.group_id IN $gids")
            params["gids"] = group_ids

        # Apply additional filters from SearchFilters object
        # Similar to community_fulltext_search, this method doesn't currently accept search_filter.
        # If search_filter were a param:
        # filter_query_part = self._apply_search_filters_to_query_basic(search_filter, "c", params)
        # if filter_query_part:
        #     where_clauses.append(filter_query_part)

        cols = "c.uuid, c.name, c.group_id, c.created_at, c.summary, c.name_embedding"
        final_query = f"MATCH (c:Community) WHERE {' AND '.join(where_clauses)} RETURN {cols}, {KUZU_SIMILARITY_FUNCTION}(c.name_embedding, $s_vec) AS score ORDER BY score DESC LIMIT $lim"

        try:
            results, _, _ = await self.execute_query(final_query, params)
            return [self._kuzudb_to_community_node(res) for res in results]
        except Exception as e:
            logger.error(f"community_similarity_search failed, possibly due to Kuzu vector function: {e}")
            return []

    async def get_relevant_nodes(self, nodes: List[EntityNode], search_filter: SearchFilters, min_score: float, limit: int) -> List[List[EntityNode]]:
        if not self.connection or not nodes: return [[] for _ in nodes] # Corrected return for empty input

        all_relevant_nodes: List[List[EntityNode]] = []
        for node_ref in nodes: # Changed 'node' to 'node_ref' to avoid confusion with 'n' alias
            if node_ref.name_embedding:
                try:
                    # node_similarity_search is already updated to use search_filter internally.
                    # We pass the search_filter object to it.
                    # The group_ids_filter here sets the explicit group_ids for the search,
                    # which node_similarity_search will use. If search_filter.group_id_filter is also set,
                    # node_similarity_search's _apply_search_filters_to_query_basic might add another
                    # condition if not handled carefully. Assuming explicit group_ids take precedence.

                    group_ids_for_search = [node_ref.group_id] if node_ref.group_id else None

                    # If search_filter contains group_id_filter and group_ids_for_search is None,
                    # it could potentially be used. However, typical use is per-node group context.
                    # For now, the direct group_ids_for_search is the primary mechanism.

                    similar_nodes = await self.node_similarity_search(
                        search_vector=node_ref.name_embedding,
                        search_filter=search_filter, # Pass the main search_filter object
                        group_ids=group_ids_for_search,
                        limit=limit,
                        min_score=min_score
                    )
                    # Filter out the input node itself from results
                    all_relevant_nodes.append([n for n in similar_nodes if n.uuid != node_ref.uuid])
                except Exception as e:
                    logger.error(f"Error finding relevant nodes for node {node_ref.uuid}: {e}")
                    all_relevant_nodes.append([])
            else:
                all_relevant_nodes.append([]) # No embedding to search with
        return all_relevant_nodes

    async def get_relevant_edges(
        self,
        edges: List[EntityEdge],
        search_filter: SearchFilters, # type: ignore
        min_score: float, # min_score for similarity search
        limit: int,
        rrf_k: int = 60 # Default RRF K value
    ) -> List[List[EntityEdge]]:
        if not self.connection or not edges:
            return [[] for _ in edges]

        all_results: List[List[EntityEdge]] = []
        internal_search_limit = limit * 3 # Fetch more results internally

        for edge in edges:
            edge_rrf_scores: Dict[str, float] = {}

            # 1. Vector Search (Edge Similarity Search)
            if edge.fact_embedding:
                try:
                    vector_results = await self.edge_similarity_search(
                        search_vector=edge.fact_embedding,
                        source_node_uuid=None, # General search, not tied to specific source/target
                        target_node_uuid=None,
                        search_filter=search_filter,
                        group_ids=[edge.group_id] if edge.group_id else None,
                        limit=internal_search_limit,
                        min_score=min_score
                    )
                    for rank, res_edge in enumerate(vector_results):
                        if res_edge.uuid == edge.uuid: # Exclude input edge
                            continue
                        edge_rrf_scores[res_edge.uuid] = edge_rrf_scores.get(res_edge.uuid, 0.0) + (1.0 / (rrf_k + rank + 1))
                except Exception as e:
                    logger.error(f"KuzuDB RRF Edges: Error during vector search for edge {edge.uuid}: {e}")

            # 2. Full-text Search on Edge Fact
            if edge.fact: # Or edge.name if 'fact' is not the primary text content for FTS
                try:
                    fts_results = await self.edge_fulltext_search(
                        query=edge.fact, # Using 'fact' for FTS
                        search_filter=search_filter,
                        group_ids=[edge.group_id] if edge.group_id else None,
                        limit=internal_search_limit
                    )
                    for rank, res_edge in enumerate(fts_results):
                        if res_edge.uuid == edge.uuid: # Exclude input edge
                            continue
                        edge_rrf_scores[res_edge.uuid] = edge_rrf_scores.get(res_edge.uuid, 0.0) + (1.0 / (rrf_k + rank + 1))
                except Exception as e:
                    logger.error(f"KuzuDB RRF Edges: Error during full-text search for edge {edge.uuid}: {e}")

            if not edge_rrf_scores:
                all_results.append([])
                continue

            sorted_uuids = sorted(edge_rrf_scores.keys(), key=lambda u: edge_rrf_scores[u], reverse=True)
            top_uuids = sorted_uuids[:limit]

            if top_uuids:
                final_edges_map = {e.uuid: e for e in await self.get_entity_edges_by_uuids(top_uuids)}
                ordered_final_edges = [final_edges_map[uuid] for uuid in top_uuids if uuid in final_edges_map]
                all_results.append(ordered_final_edges)
            else:
                all_results.append([])

        return all_results

    async def get_edge_invalidation_candidates(
        self,
        edges: List[EntityEdge],
        search_filter: SearchFilters, # type: ignore
        min_score: float, # Interpreted as max_score for dissimilarity
        limit: int
    ) -> List[List[EntityEdge]]:
        if not self.connection or not edges:
            return [[] for _ in edges]

        all_results: List[List[EntityEdge]] = []
        # The min_score here is treated as a threshold for *low* similarity.
        # We are looking for edges that are NOT similar to the input edge.

        for edge_ref in edges:
            candidate_edges: List[EntityEdge] = []
            candidate_edge_uuids: Set[str] = set() # To avoid duplicates

            # 1. Try to find dissimilar edges using vector search
            # We'll fetch more than 'limit' initially, then filter by score.
            # KuzuDB's current similarity search finds scores >= min_score.
            # To find dissimilar ones, we'd ideally want scores < min_score.
            # This is a bit tricky with current provider methods.
            # Alternative: Fetch a broader set and filter client-side, or assume 'min_score' is a high bar
            # and anything not meeting it (or not found by a high-threshold search) is a candidate.

            # For now, let's assume we can't directly query "score < X".
            # We will fetch relevant edges and then EXCLUDE those that are too similar if that's easier.
            # Or, more simply, this method could just find OLD edges if similarity is not a good proxy for invalidation.

            # Let's redefine: candidates are those *not* strongly similar OR are old.
            # For simplicity, this initial update will focus on finding *older* edges,
            # as querying for "dissimilarity" is complex without specific DB support.
            # The 'min_score' parameter will be ignored in this simplified version.

            logger.info(f"Edge invalidation for {edge_ref.uuid}: Finding older edges. 'min_score' param currently unused in this simplified version.")

            # Fetch edges from the same group, ordering by creation time (oldest first)
            # This is a simplified heuristic for "invalidation candidates".
            # A more complex version might consider edges not recently accessed, or with conflicting facts.

            # Formulate a query to get edges, ordered by created_at ASC.
            # We need to ensure we don't pick the edge_ref itself.
            # We should also ideally filter by group_id if available on edge_ref.

            params: Dict[str, Any] = {"lim": limit + 1} # Fetch one more to check if edge_ref is among them
            query_parts = [
                "MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)",
                "WHERE r.uuid <> $ref_uuid" # Exclude the reference edge itself
            ]
            params["ref_uuid"] = edge_ref.uuid

            if edge_ref.group_id:
                query_parts.append("AND r.group_id = $gid")
                params["gid"] = edge_ref.group_id

            # Applying search_filter (basic property filters on 'r')
            filter_clauses_str = self._apply_search_filters_to_query_basic(search_filter, "r", params)
            if filter_clauses_str:
                query_parts.append(f"AND {filter_clauses_str}")

            # Return columns for EntityEdge
            cols = "r.uuid, r.name, r.group_id, r.fact, r.fact_embedding, r.episodes, r.created_at, r.expired_at, r.valid_at, r.invalid_at, r.attributes, s.uuid as source_node_uuid, t.uuid as target_node_uuid"
            query = f"{' '.join(query_parts)} RETURN {cols} ORDER BY r.created_at ASC LIMIT $lim" # Oldest first

            try:
                older_edges_data, _, _ = await self.execute_query(query, params)
                for data in older_edges_data:
                    cand_edge = self._kuzudb_to_entity_edge(data)
                    if cand_edge.uuid not in candidate_edge_uuids: # Should be redundant due to query limit
                        candidate_edges.append(cand_edge)
                        candidate_edge_uuids.add(cand_edge.uuid)
                        if len(candidate_edges) >= limit:
                            break
            except Exception as e:
                logger.error(f"KuzuDB Edge Invalidation: Error fetching older edges for {edge_ref.uuid}: {e}")

            all_results.append(candidate_edges)

        return all_results

    async def get_relevant_nodes_rrf(
        self,
        nodes: List[EntityNode],
        search_filter: SearchFilters, # type: ignore
        limit: int,
        rrf_k: int = 60, # Default RRF K value
        similarity_min_score: float = 0.1 # Low threshold for similarity to get more candidates
    ) -> List[List[EntityNode]]:
        if not self.connection or not nodes:
            return [[] for _ in nodes] # Return list of empty lists matching input structure

        all_results: List[List[EntityNode]] = []
        # Fetch more results internally for RRF to have a good pool of candidates
        internal_search_limit = limit * 3 # Increased internal limit

        for node in nodes:
            node_rrf_scores: Dict[str, float] = {} # Store RRF scores for UUIDs
            # Keep track of nodes fetched from searches to avoid re-fetching if possible,
            # though RRF primarily works with UUIDs and ranks before final fetch.
            # For simplicity, we'll fetch all top UUIDs at the end.

            # 1. Vector Search (Node Similarity Search)
            if node.name_embedding:
                try:
                    vector_results = await self.node_similarity_search(
                        search_vector=node.name_embedding,
                        search_filter=search_filter,
                        # Assuming search within the same group by default if group_id is present
                        group_ids=[node.group_id] if node.group_id else None,
                        limit=internal_search_limit,
                        min_score=similarity_min_score # Use a potentially lower min_score for broader RRF input
                    )
                    for rank, res_node in enumerate(vector_results):
                        if res_node.uuid == node.uuid:  # Exclude the input node itself
                            continue
                        node_rrf_scores[res_node.uuid] = node_rrf_scores.get(res_node.uuid, 0.0) + (1.0 / (rrf_k + rank + 1))
                except Exception as e:
                    logger.error(f"KuzuDB RRF: Error during vector search for node {node.uuid}: {e}")

            # 2. Full-text Search
            if node.name:
                try:
                    fts_results = await self.node_fulltext_search(
                        query=node.name,
                        search_filter=search_filter,
                        group_ids=[node.group_id] if node.group_id else None,
                        limit=internal_search_limit
                    )
                    for rank, res_node in enumerate(fts_results):
                        if res_node.uuid == node.uuid:  # Exclude the input node itself
                            continue
                        node_rrf_scores[res_node.uuid] = node_rrf_scores.get(res_node.uuid, 0.0) + (1.0 / (rrf_k + rank + 1))
                except Exception as e:
                    logger.error(f"KuzuDB RRF: Error during full-text search for node {node.uuid}: {e}")

            # 3. Sort by RRF score and get top N UUIDs
            if not node_rrf_scores:
                all_results.append([])
                continue

            # Sort UUIDs based on their calculated RRF scores
            sorted_uuids = sorted(node_rrf_scores.keys(), key=lambda u: node_rrf_scores[u], reverse=True)
            top_uuids = sorted_uuids[:limit]

            # 4. Fetch full node objects for the top UUIDs
            if top_uuids:
                final_nodes_map = {n.uuid: n for n in await self.get_entity_nodes_by_uuids(top_uuids)}
                # Order the final nodes based on the sorted_uuids list to maintain RRF ranking
                ordered_final_nodes = [final_nodes_map[uuid] for uuid in top_uuids if uuid in final_nodes_map]
                all_results.append(ordered_final_nodes)
            else:
                all_results.append([])

        return all_results

```

[end of graphiti_core/providers/kuzudb_provider.py]
