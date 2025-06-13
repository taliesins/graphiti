# GraphDatabaseProvider Abstract Base Class

The `GraphDatabaseProvider` is an abstract base class (ABC) that defines a standardized interface for interacting with various graph database backends within the Graphiti Core framework. It ensures that different graph databases can be used interchangeably by higher-level services, provided that a concrete implementation of this interface exists for the target database.

This ABC outlines a comprehensive set of asynchronous methods for graph operations, including connection management, schema setup, CRUD (Create, Read, Update, Delete) operations for nodes and edges, bulk data loading, raw query execution, and various search functionalities.

## Connection Management

### `async def connect(self, **kwargs) -> None`
Establishes a connection to the graph database.
- **kwargs**: Provider-specific arguments required for connection (e.g., connection URI, credentials, timeouts).
- **Behavior**: Implementations should initialize and verify a connection to the database. Should be idempotent if a connection already exists.

### `async def close(self) -> None`
Closes the connection to the graph database.
- **Behavior**: Releases any resources held by the connection. Should be safe to call multiple times or on an already closed connection.

### `def get_session(self) -> Any`
Returns a database session or connection object that can be used for operations.
- **Returns**: A provider-specific session/connection object.
- **Behavior**: Should raise an error if called before `connect()` or if the connection is not active.

### `async def verify_connectivity(self) -> None`
Verifies that the connection to the database is active and operational.
- **Behavior**: Should execute a simple, non-destructive query or command to check the database status. Raises a `ConnectionError` (or provider-specific equivalent) if connectivity fails.

## Schema Management

### `async def build_indices_and_constraints(self, delete_existing: bool = False) -> None`
Builds necessary indices, constraints, and schema structures (like tables or node/relationship types) in the database required for optimal operation.
- **`delete_existing` (bool)**: If `True`, implementations should attempt to drop existing indices/constraints before recreating them. Defaults to `False`.
- **Behavior**: Ensures the database schema is correctly set up for the types of nodes, edges, and properties used by Graphiti Core.

## CRUD Operations for Nodes

### `async def save_episodic_node(self, node: EpisodicNode) -> Any`
Saves (creates or updates) an `EpisodicNode` in the database.
- **`node` (EpisodicNode)**: The episodic node object to save.
- **Returns**: Provider-specific confirmation or identifier (e.g., node UUID or database ID).
- **Behavior**: Should perform an upsert operation based on the node's UUID.

### `async def save_entity_node(self, node: EntityNode) -> Any`
Saves (creates or updates) an `EntityNode`.
- **`node` (EntityNode)**: The entity node object to save.
- **Returns**: Provider-specific confirmation or identifier.
- **Behavior**: Upsert based on UUID. Handles properties, labels, and embeddings.

### `async def save_community_node(self, node: CommunityNode) -> Any`
Saves (creates or updates) a `CommunityNode`.
- **`node` (CommunityNode)**: The community node object to save.
- **Returns**: Provider-specific confirmation or identifier.
- **Behavior**: Upsert based on UUID.

### `async def get_episodic_node_by_uuid(self, uuid: str) -> Optional[EpisodicNode]`
Retrieves an `EpisodicNode` by its unique identifier.
- **`uuid` (str)**: The UUID of the node to retrieve.
- **Returns**: An `EpisodicNode` object if found, otherwise `None`.

### `async def get_entity_node_by_uuid(self, uuid: str) -> Optional[EntityNode]`
Retrieves an `EntityNode` by its UUID.
- **`uuid` (str)**: The UUID of the node.
- **Returns**: An `EntityNode` object if found, `None` otherwise.

### `async def get_community_node_by_uuid(self, uuid: str) -> Optional[CommunityNode]`
Retrieves a `CommunityNode` by its UUID.
- **`uuid` (str)**: The UUID of the node.
- **Returns**: A `CommunityNode` object if found, `None` otherwise.

### `async def get_episodic_nodes_by_uuids(self, uuids: List[str]) -> List[EpisodicNode]`
Retrieves multiple `EpisodicNode` objects based on a list of UUIDs.
- **`uuids` (List[str])**: A list of UUIDs.
- **Returns**: A list of found `EpisodicNode` objects. Nodes not found are omitted.

### `async def get_entity_nodes_by_uuids(self, uuids: List[str]) -> List[EntityNode]`
Retrieves multiple `EntityNode` objects by their UUIDs.
- **`uuids` (List[str])**: A list of UUIDs.
- **Returns**: A list of found `EntityNode` objects.

### `async def get_community_nodes_by_uuids(self, uuids: List[str]) -> List[CommunityNode]`
Retrieves multiple `CommunityNode` objects by their UUIDs.
- **`uuids` (List[str])**: A list of UUIDs.
- **Returns**: A list of found `CommunityNode` objects.

### `async def get_episodic_nodes_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EpisodicNode]`
Retrieves `EpisodicNode` objects belonging to one or more specified `group_ids`.
- **`group_ids` (List[str])**: List of group IDs to filter by.
- **`limit` (Optional[int])**: Maximum number of nodes to return.
- **`uuid_cursor` (Optional[str])**: A UUID to be used as a cursor for pagination (e.g., fetch items created before/after this UUID, depending on sort order).
- **Returns**: A list of `EpisodicNode` objects.

### `async def get_entity_nodes_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EntityNode]`
Retrieves `EntityNode` objects by `group_ids`.
- **Parameters**: Same as `get_episodic_nodes_by_group_ids`.
- **Returns**: A list of `EntityNode` objects.

### `async def get_community_nodes_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[CommunityNode]`
Retrieves `CommunityNode` objects by `group_ids`.
- **Parameters**: Same as `get_episodic_nodes_by_group_ids`.
- **Returns**: A list of `CommunityNode` objects.

### `async def get_episodic_nodes_by_entity_node_uuid(self, entity_node_uuid: str) -> List[EpisodicNode]`
Retrieves all `EpisodicNode` objects that are connected to a specified `EntityNode` (e.g., via a `MENTIONS` relationship).
- **`entity_node_uuid` (str)**: The UUID of the `EntityNode`.
- **Returns**: A list of connected `EpisodicNode` objects.

### `async def delete_node(self, uuid: str) -> None`
Deletes a node (of any type: Entity, Episodic, or Community) from the database by its UUID.
- **`uuid` (str)**: The UUID of the node to delete.
- **Behavior**: Implementations must handle the deletion of the node and potentially its connected relationships (e.g., using `DETACH DELETE` in Cypher).

### `async def delete_nodes_by_group_id(self, group_id: str, node_type: str) -> None`
Deletes all nodes of a specified `node_type` that belong to a given `group_id`.
- **`group_id` (str)**: The group ID to filter nodes by.
- **`node_type` (str)**: The type of nodes to delete (e.g., "Entity", "Episodic", "Community").
- **Behavior**: Similar to `delete_node`, should handle detachment and deletion.

## CRUD Operations for Edges

### `async def save_episodic_edge(self, edge: EpisodicEdge) -> Any`
Saves (creates or updates) an `EpisodicEdge` (e.g., `MENTIONS` relationship between an `EpisodicNode` and an `EntityNode`).
- **`edge` (EpisodicEdge)**: The edge object to save.
- **Returns**: Provider-specific confirmation or identifier.
- **Behavior**: Upsert based on UUID.

### `async def save_entity_edge(self, edge: EntityEdge) -> Any`
Saves (creates or updates) an `EntityEdge` (e.g., `RELATES_TO` relationship between two `EntityNode` objects).
- **`edge` (EntityEdge)**: The edge object to save.
- **Returns**: Provider-specific confirmation or identifier.
- **Behavior**: Upsert based on UUID. Handles properties and embeddings.

### `async def save_community_edge(self, edge: CommunityEdge) -> Any`
Saves (creates or updates) a `CommunityEdge` (e.g., `HAS_MEMBER` relationship between a `CommunityNode` and an `EntityNode`).
- **`edge` (CommunityEdge)**: The edge object to save.
- **Returns**: Provider-specific confirmation or identifier.
- **Behavior**: Upsert based on UUID.

### `async def get_episodic_edge_by_uuid(self, uuid: str) -> Optional[EpisodicEdge]`
Retrieves an `EpisodicEdge` by its UUID.
- **`uuid` (str)**: The UUID of the edge.
- **Returns**: An `EpisodicEdge` object if found, `None` otherwise.

### `async def get_entity_edge_by_uuid(self, uuid: str) -> Optional[EntityEdge]`
Retrieves an `EntityEdge` by its UUID.
- **`uuid` (str)**: The UUID of the edge.
- **Returns**: An `EntityEdge` object if found, `None` otherwise.

### `async def get_community_edge_by_uuid(self, uuid: str) -> Optional[CommunityEdge]`
Retrieves a `CommunityEdge` by its UUID.
- **`uuid` (str)**: The UUID of the edge.
- **Returns**: A `CommunityEdge` object if found, `None` otherwise.

### `async def get_episodic_edges_by_uuids(self, uuids: List[str]) -> List[EpisodicEdge]`
Retrieves multiple `EpisodicEdge` objects by their UUIDs.
- **`uuids` (List[str])**: List of edge UUIDs.
- **Returns**: A list of found `EpisodicEdge` objects.

### `async def get_entity_edges_by_uuids(self, uuids: List[str]) -> List[EntityEdge]`
Retrieves multiple `EntityEdge` objects by their UUIDs.
- **`uuids` (List[str])**: List of edge UUIDs.
- **Returns**: A list of found `EntityEdge` objects.

### `async def get_community_edges_by_uuids(self, uuids: List[str]) -> List[CommunityEdge]`
Retrieves multiple `CommunityEdge` objects by their UUIDs.
- **`uuids` (List[str])**: List of edge UUIDs.
- **Returns**: A list of found `CommunityEdge` objects.

### `async def get_episodic_edges_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EpisodicEdge]`
Retrieves `EpisodicEdge` objects by `group_ids`.
- **Parameters**: Similar to `get_episodic_nodes_by_group_ids`.
- **Returns**: A list of `EpisodicEdge` objects.

### `async def get_entity_edges_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EntityEdge]`
Retrieves `EntityEdge` objects by `group_ids`.
- **Parameters**: Similar to `get_episodic_nodes_by_group_ids`.
- **Returns**: A list of `EntityEdge` objects.

### `async def get_community_edges_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[CommunityEdge]`
Retrieves `CommunityEdge` objects by `group_ids`.
- **Parameters**: Similar to `get_episodic_nodes_by_group_ids`.
- **Returns**: A list of `CommunityEdge` objects.

### `async def get_entity_edges_by_node_uuid(self, node_uuid: str) -> List[EntityEdge]`
Retrieves all `EntityEdge` objects connected to a given node UUID (i.e., where the node is either a source or a target).
- **`node_uuid` (str)**: The UUID of the node.
- **Returns**: A list of connected `EntityEdge` objects.

### `async def delete_edge(self, uuid: str) -> None`
Deletes an edge (of any type) from the database by its UUID.
- **`uuid` (str)**: The UUID of the edge to delete.

## Bulk Operations

### `async def add_nodes_and_edges_bulk(self, episodic_nodes: List[EpisodicNode] = None, episodic_edges: List[EpisodicEdge] = None, entity_nodes: List[EntityNode] = None, entity_edges: List[EntityEdge] = None, community_nodes: List[CommunityNode] = None, community_edges: List[CommunityEdge] = None, embedder: EmbedderClient = None) -> None`
Adds multiple nodes and edges to the database in a bulk operation for efficiency.
- **`episodic_nodes` (List[EpisodicNode])**: List of episodic nodes to add. Defaults to empty list if `None`.
- **`episodic_edges` (List[EpisodicEdge])**: List of episodic edges to add. Defaults to empty list if `None`.
- **`entity_nodes` (List[EntityNode])**: List of entity nodes to add. Defaults to empty list if `None`.
- **`entity_edges` (List[EntityEdge])**: List of entity edges to add. Defaults to empty list if `None`.
- **`community_nodes` (List[CommunityNode])**: List of community nodes to add. Defaults to empty list if `None`.
- **`community_edges` (List[CommunityEdge])**: List of community edges to add. Defaults to empty list if `None`.
- **`embedder` (EmbedderClient)**: An instance of an embedder client to generate embeddings for nodes/edges if they are missing.
- **Behavior**: Implementations should optimize for bulk loading (e.g., using transactions, batching, or specific bulk import tools like `COPY FROM` in KuzuDB or `UNWIND` in Neo4j). Embeddings should be generated and added to the objects before saving if not already present.

## Query Execution

### `async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]`
Executes a raw, provider-specific query against the database.
- **`query` (str)**: The query string to execute.
- **`params` (Optional[Dict[str, Any]])**: A dictionary of parameters to bind to the query.
- **`kwargs`**: Additional provider-specific options for query execution.
- **Returns**: A tuple `(records, summary_metadata, summary_counters)`.
    - `records` (List[Dict[str, Any]]): A list of result records, where each record is a dictionary.
    - `summary_metadata` (List[str]): Column names or similar metadata. (Exact structure might vary based on provider).
    - `summary_counters` (Dict[str, Any]): Statistics about the query's execution (e.g., nodes created, properties set).
- **Behavior**: This method provides an escape hatch for executing queries not covered by other ABC methods. Use with caution as it ties code to a specific database dialect.

## Search Operations

### `async def edge_fulltext_search(self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[EntityEdge]`
Performs a full-text search on `EntityEdge` objects.
- **`query` (str)**: The search query string.
- **`search_filter` (SearchFilters)**: Filters to apply to the search (e.g., date ranges, labels, types).
- **`group_ids` (Optional[List[str]])**: Optional list of group IDs to restrict the search.
- **`limit` (int)**: Maximum number of edges to return.
- **Returns**: A list of matching `EntityEdge` objects.

### `async def edge_similarity_search(self, search_vector: List[float], source_node_uuid: Optional[str], target_node_uuid: Optional[str], search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10, min_score: float = 0.6) -> List[EntityEdge]`
Performs a similarity search on `EntityEdge` objects based on a provided vector.
- **`search_vector` (List[float])**: The embedding vector to search with.
- **`source_node_uuid` (Optional[str])**: Optional UUID of the source node to narrow the search.
- **`target_node_uuid` (Optional[str])**: Optional UUID of the target node to narrow the search.
- **`search_filter` (SearchFilters)**: Additional filters.
- **`group_ids` (Optional[List[str]])**: Optional group ID filter.
- **`limit` (int)**: Maximum number of edges.
- **`min_score` (float)**: Minimum similarity score for an edge to be considered a match.
- **Returns**: A list of matching `EntityEdge` objects, typically ordered by similarity score.

### `async def edge_bfs_search(self, bfs_origin_node_uuids: Optional[List[str]], bfs_max_depth: int, search_filter: SearchFilters, limit: int) -> List[EntityEdge]`
Performs a Breadth-First Search (BFS) starting from `bfs_origin_node_uuids` to find `EntityEdge` objects.
- **`bfs_origin_node_uuids` (Optional[List[str]])**: List of UUIDs of origin nodes for the BFS.
- **`bfs_max_depth` (int)**: Maximum depth of the BFS traversal.
- **`search_filter` (SearchFilters)**: Filters to apply to the edges found during traversal (e.g., relationship types, properties).
- **`limit` (int)**: Maximum number of edges to return.
- **Returns**: A list of `EntityEdge` objects found by the BFS.

### `async def node_fulltext_search(self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[EntityNode]`
Performs a full-text search on `EntityNode` objects.
- **Parameters**: Similar to `edge_fulltext_search`.
- **Returns**: A list of matching `EntityNode` objects.

### `async def node_similarity_search(self, search_vector: List[float], search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10, min_score: float = 0.6) -> List[EntityNode]`
Performs a similarity search on `EntityNode` objects.
- **Parameters**: Similar to `edge_similarity_search` (without source/target node UUIDs).
- **Returns**: A list of matching `EntityNode` objects.

### `async def node_bfs_search(self, bfs_origin_node_uuids: Optional[List[str]], search_filter: SearchFilters, bfs_max_depth: int, limit: int) -> List[EntityNode]`
Performs a BFS starting from `bfs_origin_node_uuids` to find `EntityNode` objects.
- **Parameters**: Similar to `edge_bfs_search`.
- **`search_filter`**: Filters apply to the *target nodes* found, and potentially to edges in the path.
- **Returns**: A list of `EntityNode` objects found.

### `async def episode_fulltext_search(self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[EpisodicNode]`
Performs a full-text search on `EpisodicNode` objects.
- **Parameters**: Similar to `node_fulltext_search`.
- **Returns**: A list of matching `EpisodicNode` objects.

### `async def community_fulltext_search(self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[CommunityNode]`
Performs a full-text search on `CommunityNode` objects.
- **Parameters**: Similar to `node_fulltext_search`.
- **Returns**: A list of matching `CommunityNode` objects.

### `async def community_similarity_search(self, search_vector: List[float], search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10, min_score: float = 0.6) -> List[CommunityNode]`
Performs a similarity search on `CommunityNode` objects.
- **Parameters**: Similar to `node_similarity_search`.
- **Returns**: A list of matching `CommunityNode` objects.

## Embedding Retrieval

### `async def get_embeddings_for_nodes(self, nodes: List[EntityNode]) -> Dict[str, List[float]]`
Retrieves stored name embeddings for a list of `EntityNode` objects.
- **`nodes` (List[EntityNode])**: A list of `EntityNode` objects for which to fetch embeddings.
- **Returns**: A dictionary mapping node UUIDs to their embedding vectors (List[float]). Nodes without embeddings or not found are omitted.

### `async def get_embeddings_for_communities(self, communities: List[CommunityNode]) -> Dict[str, List[float]]`
Retrieves stored name embeddings for `CommunityNode` objects.
- **`communities` (List[CommunityNode])**: List of `CommunityNode` objects.
- **Returns**: Dictionary mapping community UUID to embedding vector.

### `async def get_embeddings_for_edges(self, edges: List[EntityEdge]) -> Dict[str, List[float]]`
Retrieves stored fact embeddings for `EntityEdge` objects.
- **`edges` (List[EntityEdge])**: List of `EntityEdge` objects.
- **Returns**: Dictionary mapping edge UUID to embedding vector.

## Graph Traversal and Relationship Queries

### `async def get_mentioned_nodes(self, episodes: List[EpisodicNode]) -> List[EntityNode]`
Retrieves `EntityNode` objects that are mentioned in (connected to) a given list of `EpisodicNode` objects.
- **`episodes` (List[EpisodicNode])**: List of `EpisodicNode` objects.
- **Returns**: A list of unique `EntityNode` objects mentioned.

### `async def get_communities_by_nodes(self, nodes: List[EntityNode]) -> List[CommunityNode]`
Retrieves `CommunityNode` objects that are associated with (e.g., have as members) a given list of `EntityNode` objects.
- **`nodes` (List[EntityNode])**: List of `EntityNode` objects.
- **Returns**: A list of unique `CommunityNode` objects.

## Relevance and Candidate Finding

### `async def get_relevant_nodes(self, nodes: List[EntityNode], search_filter: SearchFilters, min_score: float, limit: int) -> List[List[EntityNode]]`
Finds relevant `EntityNode` objects for each node in an input list. Relevance can be determined by similarity, shared connections, or other provider-specific logic (often RRF).
- **`nodes` (List[EntityNode])**: A list of input `EntityNode` objects.
- **`search_filter` (SearchFilters)**: Filters to apply during the relevance search.
- **`min_score` (float)**: A minimum score threshold for relevance (interpretation depends on provider).
- **`limit` (int)**: Maximum number of relevant nodes to return per input node.
- **Returns**: A list of lists, where each inner list contains relevant `EntityNode` objects for the corresponding input node.

### `async def get_relevant_edges(self, edges: List[EntityEdge], search_filter: SearchFilters, min_score: float, limit: int) -> List[List[EntityEdge]]`
Finds relevant `EntityEdge` objects for each edge in an input list.
- **Parameters**: Similar to `get_relevant_nodes`, but for edges.
- **Returns**: A list of lists of relevant `EntityEdge` objects.

### `async def get_edge_invalidation_candidates(self, edges: List[EntityEdge], search_filter: SearchFilters, min_score: float, limit: int) -> List[List[EntityEdge]]`
Identifies `EntityEdge` objects that are candidates for invalidation based on similarity or other criteria relative to a list of input edges. This is often used to find potentially redundant or conflicting edges.
- **`edges` (List[EntityEdge])**: List of reference `EntityEdge` objects.
- **`search_filter` (SearchFilters)**: Filters for the candidate search.
- **`min_score` (float)**: Minimum similarity/relevance score for an edge to be considered a candidate.
- **`limit` (int)**: Maximum number of candidates per input edge.
- **Returns**: A list of lists of candidate `EntityEdge` objects.

## Maintenance and Utility

### `async def clear_data(self, group_ids: Optional[List[str]] = None) -> None`
Clears data from the database.
- **`group_ids` (Optional[List[str]])**: If provided, only data related to these group IDs will be cleared.
- **Behavior**: If `group_ids` is `None`, the behavior is provider-dependent: it might clear all graph data or raise an error requiring explicit group IDs. Implementations should clearly document their behavior for a `None` `group_ids` argument.

### `async def retrieve_episodes(self, reference_time: datetime, last_n: int, group_ids: Optional[List[str]] = None, source: Optional[EpisodeType] = None) -> List[EpisodicNode]`
Retrieves the `last_n` most recent `EpisodicNode` objects created at or before a `reference_time`.
- **`reference_time` (datetime)**: The point in time to query relative to (typically `datetime.now(timezone.utc)`).
- **`last_n` (int)**: The number of most recent episodes to retrieve.
- **`group_ids` (Optional[List[str]])**: Optional list of group IDs to filter episodes.
- **`source` (Optional[EpisodeType])**: Optional `EpisodeType` to filter episodes.
- **Returns**: A list of `EpisodicNode` objects, usually sorted from oldest to newest among the `last_n`.

### `async def count_node_mentions(self, node_uuid: str) -> int`
Counts how many distinct `EpisodicNode` objects mention (are connected to via a `MENTIONS` relationship) a given `EntityNode`.
- **`node_uuid` (str)**: The UUID of the `EntityNode`.
- **Returns**: The number of distinct mentioning episodes.
