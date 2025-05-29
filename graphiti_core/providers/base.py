from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple

# Assuming these types exist in graphiti_core.nodes and graphiti_core.edges
# and other relevant modules.
# If they are in different locations, the import paths might need adjustment.
try:
    from graphiti_core.nodes import EpisodicNode, EntityNode, CommunityNode
    from graphiti_core.edges import EpisodicEdge, EntityEdge, CommunityEdge
    # Assuming EmbedderClient, SearchFilters, EpisodeType are defined elsewhere
    # For example, in graphiti_core.search or graphiti_core.utils
    from graphiti_core.embedder import EmbedderClient # Placeholder
    from graphiti_core.search_filter import SearchFilters # Placeholder
    from graphiti_core.episode_type import EpisodeType # Placeholder
except ImportError:
    # Fallback for environments where these types might not be immediately available
    # This helps in defining the interface even if concrete types are in flux
    EpisodicNode = Any
    EntityNode = Any
    CommunityNode = Any
    EpisodicEdge = Any
    EntityEdge = Any
    CommunityEdge = Any
    EmbedderClient = Any
    SearchFilters = Any
    EpisodeType = Any


class GraphDatabaseProvider(ABC):
    """
    Abstract base class for graph database providers.
    Defines the interface for interacting with different graph database backends.
    """

    # Connection/Session Management
    @abstractmethod
    async def connect(self, **kwargs) -> None:
        """Establishes a connection to the graph database."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Closes the connection to the graph database."""
        pass

    @abstractmethod
    def get_session(self) -> Any:
        """
        Returns a database session or connection object.
        The exact type may be provider-specific or a wrapper.
        """
        pass

    # Index and Constraints
    @abstractmethod
    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        """
        Builds necessary indices and constraints in the database.
        :param delete_existing: If True, existing indices/constraints might be dropped and recreated.
        """
        pass

    # CRUD Operations for Nodes
    @abstractmethod
    async def save_episodic_node(self, node: EpisodicNode) -> Any:
        """Saves an episodic node to the database."""
        pass

    @abstractmethod
    async def save_entity_node(self, node: EntityNode) -> Any:
        """Saves an entity node to the database."""
        pass

    @abstractmethod
    async def save_community_node(self, node: CommunityNode) -> Any:
        """Saves a community node to the database."""
        pass

    @abstractmethod
    async def get_episodic_node_by_uuid(self, uuid: str) -> Optional[EpisodicNode]:
        """Retrieves an episodic node by its UUID."""
        pass

    @abstractmethod
    async def get_entity_node_by_uuid(self, uuid: str) -> Optional[EntityNode]:
        """Retrieves an entity node by its UUID."""
        pass

    @abstractmethod
    async def get_community_node_by_uuid(self, uuid: str) -> Optional[CommunityNode]:
        """Retrieves a community node by its UUID."""
        pass

    @abstractmethod
    async def get_episodic_nodes_by_uuids(self, uuids: List[str]) -> List[EpisodicNode]:
        """Retrieves multiple episodic nodes by their UUIDs."""
        pass

    @abstractmethod
    async def get_entity_nodes_by_uuids(self, uuids: List[str]) -> List[EntityNode]:
        """Retrieves multiple entity nodes by their UUIDs."""
        pass

    @abstractmethod
    async def get_community_nodes_by_uuids(self, uuids: List[str]) -> List[CommunityNode]:
        """Retrieves multiple community nodes by their UUIDs."""
        pass

    @abstractmethod
    async def get_episodic_nodes_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EpisodicNode]:
        """Retrieves episodic nodes belonging to specified group_ids, with optional pagination."""
        pass

    @abstractmethod
    async def get_entity_nodes_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EntityNode]:
        """Retrieves entity nodes belonging to specified group_ids, with optional pagination."""
        pass

    @abstractmethod
    async def get_community_nodes_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[CommunityNode]:
        """Retrieves community nodes belonging to specified group_ids, with optional pagination."""
        pass

    @abstractmethod
    async def get_episodic_nodes_by_entity_node_uuid(self, entity_node_uuid: str) -> List[EpisodicNode]:
        """Retrieves episodic nodes connected to a given entity node."""
        pass

    @abstractmethod
    async def delete_node(self, uuid: str) -> None:
        """
        Deletes a node (of any type) by its UUID.
        Implementers will need to handle or determine the node type if necessary.
        """
        pass

    @abstractmethod
    async def delete_nodes_by_group_id(self, group_id: str, node_type: str) -> None:
        """
        Deletes nodes of a specific type belonging to a group_id.
        :param group_id: The group ID.
        :param node_type: One of "Entity", "Episodic", "Community".
        """
        pass

    # CRUD Operations for Edges
    @abstractmethod
    async def save_episodic_edge(self, edge: EpisodicEdge) -> Any:
        """Saves an episodic edge to the database."""
        pass

    @abstractmethod
    async def save_entity_edge(self, edge: EntityEdge) -> Any:
        """Saves an entity edge to the database."""
        pass

    @abstractmethod
    async def save_community_edge(self, edge: CommunityEdge) -> Any:
        """Saves a community edge to the database."""
        pass

    @abstractmethod
    async def get_episodic_edge_by_uuid(self, uuid: str) -> Optional[EpisodicEdge]:
        """Retrieves an episodic edge by its UUID."""
        pass

    @abstractmethod
    async def get_entity_edge_by_uuid(self, uuid: str) -> Optional[EntityEdge]:
        """Retrieves an entity edge by its UUID."""
        pass

    @abstractmethod
    async def get_community_edge_by_uuid(self, uuid: str) -> Optional[CommunityEdge]:
        """Retrieves a community edge by its UUID."""
        pass

    @abstractmethod
    async def get_episodic_edges_by_uuids(self, uuids: List[str]) -> List[EpisodicEdge]:
        """Retrieves multiple episodic edges by their UUIDs."""
        pass

    @abstractmethod
    async def get_entity_edges_by_uuids(self, uuids: List[str]) -> List[EntityEdge]:
        """Retrieves multiple entity edges by their UUIDs."""
        pass

    @abstractmethod
    async def get_community_edges_by_uuids(self, uuids: List[str]) -> List[CommunityEdge]:
        """Retrieves multiple community edges by their UUIDs."""
        pass

    @abstractmethod
    async def get_episodic_edges_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EpisodicEdge]:
        """Retrieves episodic edges belonging to specified group_ids, with optional pagination."""
        pass

    @abstractmethod
    async def get_entity_edges_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[EntityEdge]:
        """Retrieves entity edges belonging to specified group_ids, with optional pagination."""
        pass

    @abstractmethod
    async def get_community_edges_by_group_ids(self, group_ids: List[str], limit: Optional[int] = None, uuid_cursor: Optional[str] = None) -> List[CommunityEdge]:
        """Retrieves community edges belonging to specified group_ids, with optional pagination."""
        pass

    @abstractmethod
    async def get_entity_edges_by_node_uuid(self, node_uuid: str) -> List[EntityEdge]:
        """Retrieves entity edges connected to a given node UUID (either source or target)."""
        pass

    @abstractmethod
    async def delete_edge(self, uuid: str) -> None:
        """
        Deletes an edge (of any type) by its UUID.
        Implementers will need to handle or determine the edge type if necessary.
        """
        pass

    # Bulk Operations
    @abstractmethod
    async def add_nodes_and_edges_bulk(
        self,
        episodic_nodes: List[EpisodicNode],
        episodic_edges: List[EpisodicEdge],
        entity_nodes: List[EntityNode],
        entity_edges: List[EntityEdge],
        embedder: EmbedderClient
    ) -> None:
        """
        Adds nodes and edges in bulk.
        This is a complex operation and might be broken down or re-thought
        for a generic interface depending on provider capabilities.
        """
        pass

    # Query Execution
    @abstractmethod
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
        """
        Executes a raw query against the database.
        Matches Neo4j driver's `execute_query` return signature for now.
        May need generalization.
        Returns: A tuple containing (records, summary_metadata, summary_counters).
        """
        pass

    # Search Operations
    @abstractmethod
    async def edge_fulltext_search(self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[EntityEdge]:
        """Performs full-text search on entity edges."""
        pass

    @abstractmethod
    async def edge_similarity_search(self, search_vector: List[float], source_node_uuid: Optional[str], target_node_uuid: Optional[str], search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10, min_score: float = 0.6) -> List[EntityEdge]:
        """Performs similarity search on entity edges based on a vector."""
        pass

    @abstractmethod
    async def edge_bfs_search(self, bfs_origin_node_uuids: Optional[List[str]], bfs_max_depth: int, search_filter: SearchFilters, limit: int) -> List[EntityEdge]:
        """Performs a Breadth-First Search for entity edges."""
        pass

    @abstractmethod
    async def node_fulltext_search(self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[EntityNode]:
        """Performs full-text search on entity nodes."""
        pass

    @abstractmethod
    async def node_similarity_search(self, search_vector: List[float], search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10, min_score: float = 0.6) -> List[EntityNode]:
        """Performs similarity search on entity nodes based on a vector."""
        pass

    @abstractmethod
    async def node_bfs_search(self, bfs_origin_node_uuids: Optional[List[str]], search_filter: SearchFilters, bfs_max_depth: int, limit: int) -> List[EntityNode]:
        """Performs a Breadth-First Search for entity nodes."""
        pass

    @abstractmethod
    async def episode_fulltext_search(self, query: str, search_filter: SearchFilters, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[EpisodicNode]:
        """Performs full-text search on episodic nodes."""
        pass

    @abstractmethod
    async def community_fulltext_search(self, query: str, group_ids: Optional[List[str]] = None, limit: int = 10) -> List[CommunityNode]:
        """Performs full-text search on community nodes."""
        pass

    @abstractmethod
    async def community_similarity_search(self, search_vector: List[float], group_ids: Optional[List[str]] = None, limit: int = 10, min_score: float = 0.6) -> List[CommunityNode]:
        """Performs similarity search on community nodes based on a vector."""
        pass

    @abstractmethod
    async def get_embeddings_for_nodes(self, nodes: List[EntityNode]) -> Dict[str, List[float]]:
        """Retrieves embeddings for a list of entity nodes."""
        pass

    @abstractmethod
    async def get_embeddings_for_communities(self, communities: List[CommunityNode]) -> Dict[str, List[float]]:
        """Retrieves embeddings for a list of community nodes."""
        pass

    @abstractmethod
    async def get_embeddings_for_edges(self, edges: List[EntityEdge]) -> Dict[str, List[float]]:
        """Retrieves embeddings for a list of entity edges."""
        pass

    @abstractmethod
    async def get_mentioned_nodes(self, episodes: List[EpisodicNode]) -> List[EntityNode]:
        """Retrieves entity nodes mentioned in a list of episodic nodes."""
        pass

    @abstractmethod
    async def get_communities_by_nodes(self, nodes: List[EntityNode]) -> List[CommunityNode]:
        """Retrieves communities associated with a list of entity nodes."""
        pass

    @abstractmethod
    async def get_relevant_nodes(self, nodes: List[EntityNode], search_filter: SearchFilters, min_score: float, limit: int) -> List[List[EntityNode]]:
        """Finds relevant entity nodes for a given list of entity nodes."""
        pass

    @abstractmethod
    async def get_relevant_edges(self, edges: List[EntityEdge], search_filter: SearchFilters, min_score: float, limit: int) -> List[List[EntityEdge]]:
        """Finds relevant entity edges for a given list of entity edges."""
        pass

    @abstractmethod
    async def get_edge_invalidation_candidates(self, edges: List[EntityEdge], search_filter: SearchFilters, min_score: float, limit: int) -> List[List[EntityEdge]]:
        """Identifies entity edges that are candidates for invalidation based on relevance or other criteria."""
        pass

    # Maintenance/Utility
    @abstractmethod
    async def clear_data(self, group_ids: Optional[List[str]] = None) -> None:
        """
        Clears data from the database.
        If group_ids is provided, only data related to these groups will be cleared.
        Otherwise, all graph data may be cleared (provider-dependent implementation).
        """
        pass

    @abstractmethod
    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int,
        group_ids: Optional[List[str]] = None,
        source: Optional[EpisodeType] = None
    ) -> List[EpisodicNode]:
        """
        Retrieves the last N episodic nodes relative to a reference time,
        optionally filtered by group_ids and source.
        """
        pass

    @abstractmethod
    async def verify_connectivity(self) -> None:
        """
        Verifies that the connection to the database is alive and operational.
        Raises an exception if connectivity cannot be verified.
        """
        pass
