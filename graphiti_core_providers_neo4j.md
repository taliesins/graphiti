# Neo4jProvider

The `Neo4jProvider` is a concrete implementation of the `GraphDatabaseProvider` abstract base class, designed to interface with a [Neo4j](https://neo4j.com/) graph database. Neo4j is a popular, enterprise-grade native graph database that uses Cypher as its query language and supports ACID transactions, indexing (including range, full-text, and vector indexes), and clustering.

This provider allows Graphiti Core to leverage Neo4j's capabilities for storing, querying, and managing complex graph data, including nodes (Entity, Episodic, Community) and their relationships.

## Constructor

### `__init__(self, uri: str, user: str, password: str, database: Optional[str] = None)`
Initializes the `Neo4jProvider`.
- **`uri` (str)**: The connection URI for the Neo4j instance (e.g., "neo4j://localhost:7687", "neo4j+s://yourinstance.databases.neo4j.io").
- **`user` (str)**: The username for Neo4j authentication.
- **`password` (str)**: The password for Neo4j authentication.
- **`database` (Optional[str])**: The name of the Neo4j database to connect to. Defaults to the Neo4j default database (usually "neo4j") if not specified or if the `DEFAULT_DATABASE` helper constant is used.

## Connection Management

### `async def connect(self, **kwargs) -> None`
Initializes the Neo4j asynchronous driver (`neo4j.AsyncGraphDatabase.driver`) using the provided URI and credentials. It also verifies connectivity to the database.

### `async def close(self) -> None`
Closes the Neo4j driver if it's active, releasing any resources.

### `def get_session(self) -> neo4j.AsyncSession`
Returns a new Neo4j `AsyncSession` object configured for the specified database. Raises a `ConnectionError` if the driver is not initialized. Sessions are lightweight and should typically be used per logical unit of work.

### `async def verify_connectivity(self) -> None`
Verifies that the connection to the database is active and operational by calling the driver's `verify_connectivity` method. Raises a `ConnectionError` if connectivity fails.

## Schema Management

### `async def build_indices_and_constraints(self, delete_existing: bool = False) -> None`
Creates a predefined set of indices in Neo4j to optimize query performance.
- **Behavior**:
    - If `delete_existing` is `True`, it first calls `SHOW INDEXES YIELD name` and then `DROP INDEX $name` for each existing index.
    - It then creates several types of indexes if they do not already exist:
        - **Range Indexes**: On `uuid` and `group_id` for `Entity`, `Episodic`, `Community` nodes, and `RELATES_TO`, `MENTIONS`, `HAS_MEMBER` relationships. Also on common properties like `name` (for `Entity`), `created_at` (for `Entity`, `Episodic`, `RELATES_TO`), `valid_at` (for `Episodic`, `RELATES_TO`), `expired_at` (for `RELATES_TO`), and `invalid_at` (for `RELATES_TO`).
        - **Full-text Indexes**:
            - `episode_content`: For `Episodic` nodes on properties `content`, `source`, `source_description`, `group_id`.
            - `node_name_and_summary`: For `Entity` nodes on `name`, `summary`, `group_id`.
            - `community_name`: For `Community` nodes on `name`, `group_id`.
            - `edge_name_and_fact`: For `RELATES_TO` relationships on `name`, `fact`, `group_id`.
        - **Vector Indexes**: While not explicitly created here via `CREATE VECTOR INDEX`, vector indexes (e.g., on `name_embedding`, `fact_embedding`) are implicitly handled by Neo4j AuraDB's vector capabilities. Node/edge save methods use `db.create.setNodeVectorProperty` and `db.create.setRelationshipVectorProperty` which integrate with AuraDB's built-in vector indexing. For self-hosted Neo4j, vector index creation would be separate.
- **Note**: Uses `semaphore_gather` from `graphiti_core.helpers` to run index creation queries concurrently for efficiency.

## CRUD Operations

Node and edge properties are typically stored directly on the Neo4j nodes/relationships. Embeddings are stored as list properties (e.g., `name_embedding`, `fact_embedding`). Attributes are merged into the node/relationship properties.

### Node Saving (`save_entity_node`, `save_episodic_node`, `save_community_node`)
- Uses Cypher `MERGE` operations based on the node's `uuid` to achieve upsert behavior.
- **`save_entity_node`**: Constructs a Cypher query to `MERGE` on `(n:Entity {uuid: $entity_data.uuid})`. It then iterates through `node.labels` (excluding 'Entity') to `SET n:\`label\`` for each additional label. Properties from `node.attributes` are merged into `entity_data`, and then relevant properties are set using `SET n += $props_to_set`. Embeddings are set using `CALL db.create.setNodeVectorProperty(n, 'name_embedding', data.name_embedding)`.
- **`save_episodic_node` / `save_community_node`**: Use predefined Cypher queries from `graphiti_core.models.nodes.node_db_queries` (e.g., `EPISODIC_NODE_SAVE`, `COMMUNITY_NODE_SAVE`). These queries also handle upserting based on UUID and setting all properties, including embeddings via Neo4j procedures if applicable (e.g., `COMMUNITY_NODE_SAVE` includes `db.create.setNodeVectorProperty`).

### Node Retrieval (`get_*_node_by_uuid`, `get_*_nodes_by_uuids`)
- Uses `MATCH (n:Label {uuid: $uuid}) RETURN ...` or `MATCH (n:Label) WHERE n.uuid IN $uuids RETURN ...`.
- Return clauses (like `ENTITY_NODE_RETURN`) are structured to fetch all necessary properties, including embeddings and labels. Helper functions like `get_entity_node_from_record` parse these Neo4j records into Pydantic models, converting Neo4j-specific data types (like `DateTime`) to Python types.

### Node Retrieval by Group ID (`get_*_nodes_by_group_ids`)
- Uses `MATCH (n:Label) WHERE n.group_id IN $group_ids ... RETURN ...`.
- **Pagination**: Implements keyset pagination using `uuid_cursor`. The Cypher query, constructed by the internal `_get_nodes_by_group_ids` helper, includes `AND alias.uuid < $uuid_cursor` (since the default sort order is `alias.uuid DESC`) when `uuid_cursor` is provided, along with `ORDER BY alias.uuid DESC LIMIT $limit`.

### Edge Saving (`save_entity_edge`, `save_episodic_edge`, `save_community_edge`)
- Uses Cypher `MATCH` for source/target nodes and `MERGE` for the relationship based on its `uuid`.
- Properties are set using `SET r = $edge_data` or similar.
- `save_entity_edge` uses `CALL db.create.setRelationshipVectorProperty(...)` for `fact_embedding`.
- Uses predefined Cypher queries from `graphiti_core.models.edges.edge_db_queries` (e.g., `ENTITY_EDGE_SAVE`).

### Edge Retrieval (`get_*_edge_by_uuid`, `get_*_edges_by_uuids`)
- Uses `MATCH (n)-[e:TYPE {uuid: $uuid}]->(m) RETURN ...` or similar with `WHERE e.uuid IN $uuids`.
- Return clauses (like `ENTITY_EDGE_RETURN`) fetch all edge properties and source/target node UUIDs. Parsed by helpers like `get_entity_edge_from_record`.

### Edge Retrieval by Group ID (`get_*_edges_by_group_ids`)
- Similar to node retrieval by group ID, uses `MATCH ()-[e:TYPE]->() WHERE e.group_id IN $group_ids ... RETURN ...`.
- **Pagination**: Implements keyset pagination using `uuid_cursor` and `ORDER BY e.uuid DESC LIMIT $limit` via the `_get_edges_by_group_ids` helper.

### Deletion (`delete_node`, `delete_nodes_by_group_id`, `delete_edge`)
- Uses Cypher `MATCH ... DETACH DELETE ...` to delete nodes/edges and their incident relationships.
- `delete_nodes_by_group_id` targets nodes of a specific type (`Entity`, `Episodic`, or `Community`) within the group.

## Bulk Operations

### `async def add_nodes_and_edges_bulk(self, episodic_nodes: List[EpisodicNode] = None, ..., community_edges: List[CommunityEdge] = None, embedder: EmbedderClient = None) -> None`
- **Method**: Performs bulk ingestion within a single Neo4j transaction (`session.execute_write`).
- **Embedding Generation**: Iterates through `entity_nodes`, `community_nodes`, and `entity_edges` *before* the transaction to generate missing embeddings using the provided `embedder`. For `CommunityNode`, text for `name_embedding` is created by combining `node.name` and `node.summary`.
- **Processing within Transaction (`_add_nodes_and_edges_bulk_tx`)**:
    - Data for each node/edge type is prepared into a list of dictionaries suitable for the bulk queries. This includes merging attributes and ensuring correct label lists for nodes.
    - `tx.run()` is called with predefined Cypher queries (e.g., `ENTITY_NODE_SAVE_BULK`, `COMMUNITY_NODE_SAVE_BULK`, `COMMUNITY_EDGE_SAVE_BULK`) from `node_db_queries` and `edge_db_queries`. These queries use `UNWIND $list_of_data AS item MERGE ... SET ...` for efficient batch processing.
- **Supported Types**: Fully supports `EpisodicNode`, `EntityNode`, `CommunityNode`, `EpisodicEdge`, `EntityEdge`, and `CommunityEdge`.

## Query Execution

### `async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]`
- Executes a raw Cypher query using `driver.execute_query`.
- **Returns**: A tuple `(records, summary_metadata, summary_counters)` directly from the Neo4j driver.
    - `records` (List[Dict[str, Any]]): A list of result records, where each record is a `neo4j.Record` (behaves like a dictionary).
    - `summary_metadata` (List[str]): Contains keys from the summary (e.g., query, parameters, server, database).
    - `summary_counters` (Dict[str, Any]): Statistics about the query's execution (e.g., nodes_created, relationships_deleted).

## Search Operations

The provider uses helper functions (`_node_search_filter_query_constructor`, `_edge_search_filter_query_constructor`, `_fulltext_lucene_query`) to construct parts of search queries based on `SearchFilters`.

- **`_fulltext_lucene_query(query, group_ids)`**:
    - Sanitizes the input `query` string for Lucene syntax using `lucene_sanitize`.
    - Prepends `group_id` filters (e.g., `(group_id:"id1" OR group_id:"id2") AND ...`) to the Lucene query if `group_ids` are provided.
    - Enforces a `MAX_QUERY_LENGTH` (32 terms) to prevent overly long Lucene queries.
- **`_node_search_filter_query_constructor(filters: SearchFilters)`**:
    - Constructs Cypher `WHERE` clause conditions for node labels. If `filters.node_labels` is `["LabelA", "LabelB"]`, it generates `AND (n:\`LabelA\` OR n:\`LabelB\`)`. Date filters are not typically handled by this constructor for nodes as they are usually part of the Lucene query for full-text search or direct Cypher for other search types.
- **`_edge_search_filter_query_constructor(filters: SearchFilters)`**:
    - Constructs Cypher `WHERE` clause conditions for:
        - Edge types (relationship names): `AND r.name IN $edge_types`.
        - Labels of source (`n`) and target (`m`) nodes: `AND (n:\`LabelA\` OR n:\`LabelB\`) AND (m:\`LabelC\` OR m:\`LabelD\`)`.
        - Date filters (`valid_at`, `invalid_at`, `created_at`, `expired_at`) on edge properties. Supports `List[List[DateFilter]]` structure, where outer lists are ORed and inner lists are ANDed (e.g., `AND ((r.created_at > $date1 AND r.created_at < $date2) OR (r.created_at = $date3))`).

- **Full-text Search (`*_fulltext_search`)**:
    - Uses `CALL db.index.fulltext.queryNodes("index_name", $lucene_query, {limit: $limit}) YIELD node AS ..., score` or `queryRelationships` for edges.
    - The `lucene_query` is generated by `_fulltext_lucene_query`.
    - Additional Cypher `WHERE` clauses (from `SearchFilters` via the constructor helpers, and explicit `group_ids` via `_group_id_filter_clause`) are appended to filter the results from the full-text index.
    - **Indexes Used**: Relies on pre-built full-text indexes like `node_name_and_summary`, `episode_content`, `community_name`, `edge_name_and_fact`.

- **Similarity Search (`*_similarity_search`)**:
    - Uses `vector.similarity.cosine(property_embedding, $search_vector) AS score`.
    - Filters results by `min_score`.
    - Applies `SearchFilters` via Cypher `WHERE` clauses generated by the constructor helpers.
    - **Indexes Used**: Relies on Neo4j's vector indexing capabilities on embedding properties (e.g., `name_embedding`, `fact_embedding`).

- **BFS Search (`*_bfs_search`)**:
    - Constructs Cypher queries using variable-length path patterns like `(origin)-[*1..max_depth]-(target)`.
    - For `node_bfs_search`, it uses a `CALL` subquery to manage path finding and filtering per origin node. It filters paths to ensure relationships are of type `RELATES_TO` or `MENTIONS` and that the target node `n` matches `origin.group_id` and other `SearchFilters`.
    - For `edge_bfs_search`, it unnests relationships from paths found via BFS and then matches these relationships specifically against `RELATES_TO` edges, applying further filters.

## Relevance and Candidate Finding

- **`get_relevant_nodes` (RRF)**:
    - For each input node, it performs both a vector similarity search (on `name_embedding`) and a full-text search (on `name` and `summary` via Lucene).
    - Candidates from both searches (up to `limit * 2` from each) are combined using Reciprocal Rank Fusion (RRF) with a configurable `rrf_k` (default 60).
    - The top `limit` nodes by RRF score are then fetched fully and returned.
- **`get_relevant_edges`**:
    - For each input edge, performs a vector similarity search based on `edge.fact_embedding` against other edges in the same `group_id`. It does *not* currently use RRF by combining with full-text search for edges.
- **`get_edge_invalidation_candidates`**:
    - For each input edge, finds other `RELATES_TO` edges within the same `group_id` that are connected to either the source or target node of the input edge.
    - Candidates are ranked by vector similarity of their `fact_embedding` to the input edge's `fact_embedding`.

## Other Methods

- **`clear_data(group_ids)`**:
    - If `group_ids` is `None`, uses `MATCH (n) DETACH DELETE n` to clear all data in the configured database.
    - If `group_ids` is provided, uses `MATCH (n) WHERE n.group_id IN $group_ids DETACH DELETE n`.
- **`retrieve_episodes(...)`**: Fetches the `last_n` most recent `EpisodicNode` objects created at or before a `reference_time`, with optional filtering by `group_ids` and `source` type. Results are ordered chronologically.
- **`count_node_mentions(node_uuid)`**: Counts distinct `EpisodicNode` objects mentioning a given `EntityNode` using `MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $node_uuid}) RETURN count(DISTINCT e)`.

## Performance Considerations

- **Indexing**: Relies heavily on the correct setup of range, full-text, and vector indexes as defined in `build_indices_and_constraints` and by Neo4j's native vector capabilities. Query performance will degrade significantly without appropriate indexes.
- **Bulk Operations**: Uses `UNWIND` with batched parameters within a single transaction, which is generally efficient for Neo4j data ingestion.
- **Complex Filters**: Highly complex `SearchFilters` (especially with many OR conditions in date filters or deeply nested label/type checks) can lead to more complex Cypher queries. Performance should be monitored for such cases.
- **Lucene Query Length**: `_fulltext_lucene_query` enforces a `MAX_QUERY_LENGTH` (default 32 terms) to prevent overly complex queries that Lucene might struggle with or that might hit Neo4j full-text query limits.
- **RRF Search Limits**: `get_relevant_nodes` fetches `limit * 2` candidates internally from each underlying search (vector, FTS) before RRF. This multiplier affects the candidate pool size and performance. `get_relevant_edges` currently uses a direct limit for its similarity search.
- **Session Management**: `get_session()` creates a new session per call. For units of work involving multiple database interactions, it's recommended to acquire a session once and pass it through or use it within a managing context.
