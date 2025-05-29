import logging
import json # For serializing/deserializing MAP attributes
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple

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
# Assuming SearchFilters and EpisodeType are accessible, adjust imports if not.
from pydantic import BaseModel # Needed for SearchFilters if defined inline, or imported

logger = logging.getLogger(__name__)

# Define typical embedding dimension, e.g., for OpenAI ada-002
DEFAULT_EMBEDDING_DIM = 1536

# Placeholder for SearchFilters if not imported - for type hinting
class SearchFilters(BaseModel): # type: ignore
    # Define fields if specific filter logic is implemented, otherwise pass is fine
    node_labels: Optional[List[str]] = None
    edge_types: Optional[List[str]] = None
    # Add other filter fields like valid_at, created_at if they will be used by Kuzu specific filters


class KuzuDBProvider(GraphDatabaseProvider):
    def __init__(self, database_path: str, in_memory: bool = False):
        self.database_path = database_path
        self.in_memory = in_memory
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
        entity_schema = f"CREATE NODE TABLE Entity (uuid STRING, name STRING, group_id STRING, created_at TIMESTAMP, summary STRING, name_embedding FLOAT[{DEFAULT_EMBEDDING_DIM}], attributes STRING, PRIMARY KEY (uuid))"
        conn.execute(entity_schema)
        episodic_schema = f"CREATE NODE TABLE Episodic (uuid STRING, name STRING, group_id STRING, source STRING, source_description STRING, content STRING, valid_at TIMESTAMP, created_at TIMESTAMP, entity_edges LIST(STRING), PRIMARY KEY (uuid))"
        conn.execute(episodic_schema)
        community_schema = f"CREATE NODE TABLE Community (uuid STRING, name STRING, group_id STRING, created_at TIMESTAMP, summary STRING, name_embedding FLOAT[{DEFAULT_EMBEDDING_DIM}], PRIMARY KEY (uuid))"
        conn.execute(community_schema)
        mentions_schema = f"CREATE REL TABLE MENTIONS (FROM Episodic TO Entity, uuid STRING, group_id STRING, created_at TIMESTAMP)"
        conn.execute(mentions_schema)
        relates_to_schema = f"CREATE REL TABLE RELATES_TO (FROM Entity TO Entity, uuid STRING, name STRING, group_id STRING, fact STRING, fact_embedding FLOAT[{DEFAULT_EMBEDDING_DIM}], episodes LIST(STRING), created_at TIMESTAMP, expired_at TIMESTAMP, valid_at TIMESTAMP, invalid_at TIMESTAMP, attributes STRING)"
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
    
    async def _save_node_generic(self, table_name: str, node_dict: Dict[str, Any], uuid: str) -> str:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        # UPSERT logic: Try to update first (MATCH SET), if not found (or Kuzu specific error), then CREATE.
        # Kuzu's Cypher might not support MERGE or complex ON MATCH/ON CREATE yet.
        # Simplified: Attempt CREATE, if PK error, attempt UPDATE.
        # This is not atomic and less efficient. Better to use Kuzu's specific UPSERT if available.
        try:
            create_query = f"CREATE (n:{table_name} $props)"
            await self.execute_query(create_query, {"props": node_dict})
        except Exception as e_create: # Catch specific Kuzu PK violation error if possible
            logger.debug(f"CREATE failed for {table_name} {uuid} (may already exist), attempting UPDATE: {e_create}")
            try:
                set_clauses = ", ".join([f"n.{key} = ${key}" for key in node_dict if key != 'uuid'])
                update_query = f"MATCH (n:{table_name} {{uuid: $uuid}}) SET {set_clauses}"
                await self.execute_query(update_query, {**node_dict, "uuid": uuid}) # Pass uuid for MATCH, others for SET
            except Exception as e_update:
                logger.error(f"UPDATE also failed for {table_name} {uuid} after CREATE failed: {e_update}")
                raise e_update # Or handle as appropriate
        return uuid

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
        # Simplified CREATE, assumes edge doesn't exist or PK error is acceptable for now
        # KuzuDB UPSERT for edges is also complex.
        try:
            query = f"""
                MATCH (s:{source_table} {{uuid: $source_uuid}}), (t:{target_table} {{uuid: $target_uuid}})
                CREATE (s)-[r:{rel_table} $props]->(t)
            """
            # Props should not include source/target_uuid for the relationship properties itself
            props_for_rel = {k: v for k, v in edge_dict.items() if k not in ['source_node_uuid', 'target_node_uuid']}
            await self.execute_query(query, {"source_uuid": edge_dict["source_node_uuid"], "target_uuid": edge_dict["target_node_uuid"], "props": props_for_rel})
        except Exception as e:
            logger.error(f"Failed to save edge {uuid} in {rel_table}: {e}")
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
    def _apply_search_filters_to_query_basic(self, query_parts: List[str], params: Dict[str, Any], search_filter: SearchFilters, node_alias: str = "n", edge_alias: Optional[str] = None):
        # This is a very basic filter application, primarily for group_id if not already handled,
        # and simple property checks. Kuzu's specific functions for dates, lists, etc., would be needed for full SearchFilter support.
        # For now, this helper is minimal.
        # Example: if search_filter.custom_property_equals and node_alias:
        #    query_parts.append(f"AND {node_alias}.someProperty = $some_prop_val")
        #    params["some_prop_val"] = search_filter.custom_property_equals
        pass


    async def node_fulltext_search(self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[EntityNode]:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        params: Dict[str, Any] = {"query_str": query, "limit_val": limit} # Kuzu CONTAINS is exact match, not substring with %
        
        # Kuzu FTS typically uses a specific index and function like `fts_main_Entity.match($query_str)`
        # Using basic CONTAINS for now as a placeholder for actual FTS capability.
        where_clauses = ["(CONTAINS(n.name, $query_str) OR CONTAINS(n.summary, $query_str))"]
        if group_ids:
            where_clauses.append("n.group_id IN $gids")
            params["gids"] = group_ids
        
        self._apply_search_filters_to_query_basic(where_clauses, params, search_filter, node_alias="n")
        
        final_query = f"MATCH (n:Entity) WHERE {' AND '.join(where_clauses)} RETURN n.uuid, n.name, n.group_id, n.created_at, n.summary, n.name_embedding, n.attributes LIMIT $limit_val"
        results, _, _ = await self.execute_query(final_query, params)
        return [self._kuzudb_to_entity_node(res) for res in results]

    async def node_similarity_search(self, search_vector: List[float], search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10, min_score: float = 0.6) -> List[EntityNode]:
        if not self.connection: raise ConnectionError("KuzuDB connection not established.")
        # Kuzu's vector similarity function might be list_similarity for dot product, or a specific cosine_similarity.
        # Assuming cosine_similarity for now as it's common. This needs to match Kuzu's actual function.
        # If an HNSW index exists, query might be different: `LOAD FROM SCAN(Entity knn_scan('name_embedding', $search_vector, $limit))`
        KUZU_SIMILARITY_FUNCTION = "cosine_similarity" # Placeholder
        
        params: Dict[str, Any] = {"s_vec": search_vector, "lim": limit, "min_s": min_score}
        where_clauses = [f"{KUZU_SIMILARITY_FUNCTION}(n.name_embedding, $s_vec) >= $min_s"]
        if group_ids:
            where_clauses.append("n.group_id IN $gids")
            params["gids"] = group_ids
        
        self._apply_search_filters_to_query_basic(where_clauses, params, search_filter, node_alias="n")

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
        params: Dict[str, Any] = {"query_str": query, "limit_val": limit}
        where_clauses = ["(CONTAINS(r.name, $query_str) OR CONTAINS(r.fact, $query_str))"]
        if group_ids:
            where_clauses.append("r.group_id IN $gids")
            params["gids"] = group_ids
        
        self._apply_search_filters_to_query_basic(where_clauses, params, search_filter, edge_alias="r")

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
    async def add_nodes_and_edges_bulk(self, episodic_nodes: List[EpisodicNode], episodic_edges: List[EpisodicEdge], entity_nodes: List[EntityNode], entity_edges: List[EntityEdge], embedder: EmbedderClient) -> None:
        raise NotImplementedError("KuzuDBProvider add_nodes_and_edges_bulk not yet implemented")
    async def edge_similarity_search(self, search_vector: List[float], source_node_uuid: Optional[str], target_node_uuid: Optional[str], search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10, min_score: float = 0.6) -> List[EntityEdge]:
        raise NotImplementedError("KuzuDBProvider edge_similarity_search not yet implemented") # Placeholder, basic one above needs review for Kuzu func
    async def edge_bfs_search(self, bfs_origin_node_uuids: Optional[List[str]], bfs_max_depth: int, search_filter: SearchFilters, limit: int) -> List[EntityEdge]:
        raise NotImplementedError("KuzuDBProvider edge_bfs_search not yet implemented")
    async def node_bfs_search(self, bfs_origin_node_uuids: Optional[List[str]], search_filter: SearchFilters, bfs_max_depth: int, limit: int) -> List[EntityNode]:
        raise NotImplementedError("KuzuDBProvider node_bfs_search not yet implemented") # Placeholder, basic one above needs review for Kuzu func
    async def episode_fulltext_search(self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[EpisodicNode]:
        raise NotImplementedError("KuzuDBProvider episode_fulltext_search not yet implemented") # Placeholder, basic one above needs review for Kuzu func
    async def community_fulltext_search(self, query: str, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[CommunityNode]:
        raise NotImplementedError("KuzuDBProvider community_fulltext_search not yet implemented")# Placeholder, basic one above needs review for Kuzu func
    async def community_similarity_search(self, search_vector: List[float], group_ids: Optional[List[str]] = None, limit: int = 10, min_score: float = 0.6) -> List[CommunityNode]:
        raise NotImplementedError("KuzuDBProvider community_similarity_search not yet implemented")# Placeholder, basic one above needs review for Kuzu func
    async def get_relevant_nodes(self, nodes: List[EntityNode], search_filter: SearchFilters, min_score: float, limit: int) -> List[List[EntityNode]]:
        raise NotImplementedError("KuzuDBProvider get_relevant_nodes not yet implemented")
    async def get_relevant_edges(self, edges: List[EntityEdge], search_filter: SearchFilters, min_score: float, limit: int) -> List[List[EntityEdge]]:
        raise NotImplementedError("KuzuDBProvider get_relevant_edges not yet implemented")
    async def get_edge_invalidation_candidates(self, edges: List[EntityEdge], search_filter: SearchFilters, min_score: float, limit: int) -> List[List[EntityEdge]]:
        raise NotImplementedError("KuzuDBProvider get_edge_invalidation_candidates not yet implemented")

    async def add_nodes_and_edges_bulk(
        self,
        episodic_nodes: List[EpisodicNode],
        episodic_edges: List[EpisodicEdge],
        entity_nodes: List[EntityNode],
        entity_edges: List[EntityEdge],
        embedder: EmbedderClient
    ) -> None:
        if not self.connection:
            raise ConnectionError("KuzuDB connection not established.")

        logger.info(f"Starting bulk add: {len(episodic_nodes)} episodic nodes, {len(entity_nodes)} entity nodes, {len(episodic_edges)} episodic edges, {len(entity_edges)} entity edges.")

        # 1. Generate Embeddings (outside transaction)
        # Entity Nodes
        for node in entity_nodes:
            if node.name_embedding is None:
                try:
                    # Assuming EntityNode has a method to generate its own embedding
                    # This method might be part of the Pydantic model itself or a utility.
                    # For now, direct call as if it exists on the model.
                    # await node.generate_name_embedding(embedder)
                    # If generate_name_embedding is not on EntityNode, call embedder directly:
                    text_to_embed = node.name.replace('\n', ' ')
                    node.name_embedding = await embedder.create(input_data=[text_to_embed]) # Assuming embedder.create returns List[List[float]]
                    if node.name_embedding and isinstance(node.name_embedding, list) and len(node.name_embedding) > 0 and isinstance(node.name_embedding[0], list): # type: ignore
                        node.name_embedding = node.name_embedding[0] # type: ignore # Take the first embedding if create returns a list of embeddings
                    logger.debug(f"Generated name embedding for EntityNode: {node.uuid}")
                except Exception as e:
                    logger.error(f"Failed to generate embedding for EntityNode {node.uuid}: {e}")
                    # Decide if to proceed without embedding or raise error

        # Entity Edges
        for edge in entity_edges:
            if edge.fact_embedding is None:
                try:
                    # await edge.generate_embedding(embedder) # Similar to EntityNode
                    text_to_embed = edge.fact.replace('\n', ' ')
                    edge.fact_embedding = await embedder.create(input_data=[text_to_embed])
                    if edge.fact_embedding and isinstance(edge.fact_embedding, list) and len(edge.fact_embedding) > 0 and isinstance(edge.fact_embedding[0], list): # type: ignore
                        edge.fact_embedding = edge.fact_embedding[0] # type: ignore
                    logger.debug(f"Generated fact embedding for EntityEdge: {edge.uuid}")
                except Exception as e:
                    logger.error(f"Failed to generate embedding for EntityEdge {edge.uuid}: {e}")

        # KuzuDB does not support explicit BEGIN TRANSACTION / COMMIT via execute() in the same way as SQL DBs for multi-statement tx.
        # Each execute call is often its own transaction.
        # For a large batch, this means many small transactions.
        # KuzuDB's COPY FROM command is the most efficient for bulk, but that requires CSV/Parquet.
        # Iterative calls to single save methods is the fallback for API-driven bulk load.
        
        # 2. Node Creation
        logger.info("Saving episodic nodes...")
        for node in episodic_nodes:
            try:
                await self.save_episodic_node(node)
            except Exception as e:
                logger.error(f"Error saving episodic node {node.uuid} during bulk operation: {e}")
                # Optionally, collect errors and continue, or re-raise
        
        logger.info("Saving entity nodes...")
        for node in entity_nodes:
            try:
                await self.save_entity_node(node)
            except Exception as e:
                logger.error(f"Error saving entity node {node.uuid} during bulk operation: {e}")

        # 3. Edge Creation
        logger.info("Saving episodic edges...")
        for edge in episodic_edges:
            try:
                await self.save_episodic_edge(edge)
            except Exception as e:
                logger.error(f"Error saving episodic edge {edge.uuid} during bulk operation: {e}")

        logger.info("Saving entity edges...")
        for edge in entity_edges:
            try:
                await self.save_entity_edge(edge)
            except Exception as e:
                logger.error(f"Error saving entity edge {edge.uuid} during bulk operation: {e}")
        
        logger.info("Bulk add_nodes_and_edges operation completed for KuzuDB (iterative saves).")
        return # Or return some summary like counts of success/failures

```
