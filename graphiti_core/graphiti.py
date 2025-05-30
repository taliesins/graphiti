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
from datetime import datetime
from time import time

from dotenv import load_dotenv
# from neo4j import AsyncGraphDatabase # No longer directly used here
from pydantic import BaseModel
from typing_extensions import LiteralString

from graphiti_core.providers.base import GraphDatabaseProvider # Import Provider
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.graphiti_types import GraphitiClients
# from graphiti_core.helpers import DEFAULT_DATABASE, semaphore_gather # DEFAULT_DATABASE no longer needed here
from graphiti_core.helpers import semaphore_gather # semaphore_gather might still be used by utils
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode # These will be refactored to use provider
from graphiti_core.search.search import SearchConfig, search # Search will be refactored later
from graphiti_core.search.search_config import DEFAULT_SEARCH_LIMIT, SearchResults
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
# search_utils functions like get_edge_invalidation_candidates, get_mentioned_nodes, get_relevant_edges
# are now methods on the provider. Calls will be updated to self.provider.*
# from graphiti_core.search.search_utils import (
#     RELEVANT_SCHEMA_LIMIT,
#     get_edge_invalidation_candidates,
#     get_mentioned_nodes,
#     get_relevant_edges,
# )
from graphiti_core.utils.bulk_utils import (
    RawEpisode,
    add_nodes_and_edges_bulk,
    dedupe_edges_bulk,
    dedupe_nodes_bulk,
    extract_edge_dates_bulk,
    extract_nodes_and_edges_bulk,
    resolve_edge_pointers,
    retrieve_previous_episodes_bulk,
)
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.community_operations import (
    build_communities,
    remove_communities,
    update_community,
)
from graphiti_core.utils.maintenance.edge_operations import (
    build_episodic_edges,
    extract_edges,
    resolve_extracted_edge,
    resolve_extracted_edges,
)
from graphiti_core.utils.maintenance.graph_data_operations import ( # These utils will be refactored later
    EPISODE_WINDOW_LEN,
    # build_indices_and_constraints, # This will be called on provider
    # retrieve_episodes, # This will be called on provider
)
from graphiti_core.utils.maintenance.node_operations import ( # These utils will be refactored later
    extract_attributes_from_nodes,
    extract_nodes,
    resolve_extracted_nodes,
)
from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types

logger = logging.getLogger(__name__)

load_dotenv()


class AddEpisodeResults(BaseModel):
    episode: EpisodicNode
    nodes: list[EntityNode]
    edges: list[EntityEdge]


class Graphiti:
    def __init__(
        self,
        provider: GraphDatabaseProvider, # Changed from uri, user, password
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        cross_encoder: CrossEncoderClient | None = None,
        store_raw_episode_content: bool = True,
    ):
        """
        Initialize a Graphiti instance.

        This constructor sets up clients for language model operations, embedding,
        and cross-encoding, and uses a provided graph database provider for all
        database interactions.

        Parameters
        ----------
        provider : GraphDatabaseProvider
            An instance of a class that implements the GraphDatabaseProvider interface.
            This provider will be used for all graph database operations.
        llm_client : LLMClient | None, optional
            An instance of LLMClient for natural language processing tasks.
            If not provided, a default OpenAIClient will be initialized.
        embedder : EmbedderClient | None, optional
            An instance of EmbedderClient for generating embeddings.
            If not provided, a default OpenAIEmbedder will be initialized.
        cross_encoder : CrossEncoderClient | None, optional
            An instance of CrossEncoderClient for reranking search results.
            If not provided, a default OpenAIRerankerClient will be initialized.
        store_raw_episode_content : bool, optional
            Whether to store the raw content of episodes. Defaults to True.
            
        Returns
        -------
        None
        """
        self.provider = provider # Store the provider instance
        # self.driver = self.provider.driver # If Graphiti class ever needs direct access to underlying driver (should be rare)
        self.store_raw_episode_content = store_raw_episode_content
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = OpenAIClient()
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = OpenAIEmbedder()
        if cross_encoder:
            self.cross_encoder = cross_encoder
        else:
            self.cross_encoder = OpenAIRerankerClient()

        self.clients = GraphitiClients(
            provider=self.provider, # Pass provider instead of driver
            llm_client=self.llm_client,
            embedder=self.embedder,
            cross_encoder=self.cross_encoder,
        )

    async def close(self):
        """
        Close the connection to the graph database via the provider.
        """
        if self.provider:
            await self.provider.close()

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        """
        Build indices and constraints in the graph database using the provider.
        """
        if self.provider:
            await self.provider.build_indices_and_constraints(delete_existing)

    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = EPISODE_WINDOW_LEN, # This default might be provider-specific or stay general
        group_ids: list[str] | None = None,
        source: EpisodeType | None = None,
    ) -> list[EpisodicNode]:
        """
        Retrieve the last n episodic nodes from the graph using the provider.
        """
        if self.provider:
            return await self.provider.retrieve_episodes(
                reference_time=reference_time,
                last_n=last_n,
                group_ids=group_ids,
                source=source,
            )
        return [] # Or raise an error if provider is not set

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        source: EpisodeType = EpisodeType.message,
        group_id: str = '',
        uuid: str | None = None,
        update_communities: bool = False,
        entity_types: dict[str, BaseModel] | None = None,
        previous_episode_uuids: list[str] | None = None,
        edge_types: dict[str, BaseModel] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,
    ) -> AddEpisodeResults:
        """
        Process an episode and update the graph.

        This method extracts information from the episode, creates nodes and edges,
        and updates the graph database accordingly.

        Parameters
        ----------
        name : str
            The name of the episode.
        episode_body : str
            The content of the episode.
        source_description : str
            A description of the episode's source.
        reference_time : datetime
            The reference time for the episode.
        source : EpisodeType, optional
            The type of the episode. Defaults to EpisodeType.message.
        group_id : str | None
            An id for the graph partition the episode is a part of.
        uuid : str | None
            Optional uuid of the episode.
        update_communities : bool
            Optional. Whether to update communities with new node information
        previous_episode_uuids : list[str] | None
            Optional.  list of episode uuids to use as the previous episodes. If this is not provided,
            the most recent episodes by created_at date will be used.

        Returns
        -------
        None

        Notes
        -----
        This method performs several steps including node extraction, edge extraction,
        deduplication, and database updates. It also handles embedding generation
        and edge invalidation.

        It is recommended to run this method as a background process, such as in a queue.
        It's important that each episode is added sequentially and awaited before adding
        the next one. For web applications, consider using FastAPI's background tasks
        or a dedicated task queue like Celery for this purpose.

        Example using FastAPI background tasks:
            @app.post("/add_episode")
            async def add_episode_endpoint(episode_data: EpisodeData):
                background_tasks.add_task(graphiti.add_episode, **episode_data.dict())
                return {"message": "Episode processing started"}
        """
        try:
            start = time()
            now = utc_now()

            validate_entity_types(entity_types)

            previous_episodes = (
                await self.retrieve_episodes(
                    reference_time, # Uses self.retrieve_episodes which is now provider-based
                    last_n=RELEVANT_SCHEMA_LIMIT,
                    group_ids=[group_id],
                    source=source,
                )
                if previous_episode_uuids is None
                # Assuming EpisodicNode.get_by_uuids was refactored to take provider
                else await self.provider.get_episodic_nodes_by_uuids(uuids=previous_episode_uuids)
            )

            episode = (
                # Assuming EpisodicNode.get_by_uuid was refactored to take provider
                await self.provider.get_episodic_node_by_uuid(uuid=uuid)
                if uuid is not None
                else EpisodicNode( # This is creating a new local object, not fetching
                    name=name,
                    group_id=group_id,
                    labels=[],
                    source=source,
                    content=episode_body,
                    source_description=source_description,
                    created_at=now,
                    valid_at=reference_time,
                )
            )

            # Create default edge type map
            edge_type_map_default = (
                {('Entity', 'Entity'): list(edge_types.keys())}
                if edge_types is not None
                else {('Entity', 'Entity'): []}
            )

            # Extract entities as nodes

            extracted_nodes = await extract_nodes(
                self.clients, episode, previous_episodes, entity_types
            )

            # Extract edges and resolve nodes
            (nodes, uuid_map), extracted_edges = await semaphore_gather(
                resolve_extracted_nodes(
                    self.clients,
                    extracted_nodes,
                    episode,
                    previous_episodes,
                    entity_types,
                ),
                extract_edges(
                    self.clients, episode, extracted_nodes, previous_episodes, group_id, edge_types
                ),
            )

            edges = resolve_edge_pointers(extracted_edges, uuid_map)

            (resolved_edges, invalidated_edges), hydrated_nodes = await semaphore_gather(
                resolve_extracted_edges(
                    self.clients,
                    edges,
                    episode,
                    nodes,
                    edge_types or {},
                    edge_type_map or edge_type_map_default,
                ),
                extract_attributes_from_nodes(
                    self.clients, nodes, episode, previous_episodes, entity_types
                ),
            )

            entity_edges = resolved_edges + invalidated_edges

            episodic_edges = build_episodic_edges(nodes, episode, now)

            episode.entity_edges = [edge.uuid for edge in entity_edges]

            if not self.store_raw_episode_content:
                episode.content = ''

            # Use provider for bulk add
            await self.provider.add_nodes_and_edges_bulk(
                episodic_nodes=[episode], 
                episodic_edges=episodic_edges, 
                entity_nodes=hydrated_nodes, 
                entity_edges=entity_edges, 
                embedder=self.embedder
            )

            # Update any communities
            if update_communities:
                await semaphore_gather(
                    *[
                        update_community(self.provider, self.llm_client, self.embedder, node) # Pass provider
                        for node in nodes
                    ]
                )
            end = time()
            logger.info(f'Completed add_episode in {(end - start) * 1000} ms')

            return AddEpisodeResults(episode=episode, nodes=nodes, edges=entity_edges)

        except Exception as e:
            raise e

    #### WIP: USE AT YOUR OWN RISK ####
    async def add_episode_bulk(self, bulk_episodes: list[RawEpisode], group_id: str = ''):
        """
        Process multiple episodes in bulk and update the graph.

        This method extracts information from multiple episodes, creates nodes and edges,
        and updates the graph database accordingly, all in a single batch operation.

        Parameters
        ----------
        bulk_episodes : list[RawEpisode]
            A list of RawEpisode objects to be processed and added to the graph.
        group_id : str | None
            An id for the graph partition the episode is a part of.

        Returns
        -------
        None

        Notes
        -----
        This method performs several steps including:
        - Saving all episodes to the database
        - Retrieving previous episode context for each new episode
        - Extracting nodes and edges from all episodes
        - Generating embeddings for nodes and edges
        - Deduplicating nodes and edges
        - Saving nodes, episodic edges, and entity edges to the knowledge graph

        This bulk operation is designed for efficiency when processing multiple episodes
        at once. However, it's important to ensure that the bulk operation doesn't
        overwhelm system resources. Consider implementing rate limiting or chunking for
        very large batches of episodes.

        Important: This method does not perform edge invalidation or date extraction steps.
        If these operations are required, use the `add_episode` method instead for each
        individual episode.
        """
        try:
            start = time()
            now = utc_now()

            episodes = [
                EpisodicNode(
                    name=episode.name,
                    labels=[],
                    source=episode.source,
                    content=episode.content,
                    source_description=episode.source_description,
                    group_id=group_id,
                    created_at=now,
                    valid_at=episode.reference_time,
                )
                for episode in bulk_episodes
            ]

            # Save all the episodes
            # EpisodicNode.save will be refactored to use provider
            await semaphore_gather(*[episode.save(self.provider) for episode in episodes])

            # Get previous episode context for each episode
            episode_pairs = await retrieve_previous_episodes_bulk(self.provider, episodes)

            # Extract all nodes and edges
            (
                extracted_nodes,
                extracted_edges,
                episodic_edges,
            ) = await extract_nodes_and_edges_bulk(self.clients, episode_pairs)

            # Generate embeddings
            await semaphore_gather(
                *[node.generate_name_embedding(self.embedder) for node in extracted_nodes],
                *[edge.generate_embedding(self.embedder) for edge in extracted_edges],
            )

            # Dedupe extracted nodes, compress extracted edges
            # dedupe_nodes_bulk now takes provider
            (nodes, uuid_map), extracted_edges_timestamped = await semaphore_gather(
                dedupe_nodes_bulk(self.provider, self.llm_client, extracted_nodes),
                extract_edge_dates_bulk(self.llm_client, extracted_edges, episode_pairs), # This is LLM only
            )

            # Instead of individual saves, collect all and use provider's bulk add.
            # The original code here was saving nodes, then episodic edges, then entity edges.
            # This can be simplified by preparing all lists and calling provider.add_nodes_and_edges_bulk once.

            # Episodic edges were already built. Entity edges need to be prepared.
            episodic_edges_final = resolve_edge_pointers(episodic_edges, uuid_map)
            entity_edges_final = await dedupe_edges_bulk(
                self.provider, self.llm_client, resolve_edge_pointers(extracted_edges_timestamped, uuid_map)
            )

            # Note: Updating individual 'episode.entity_edges' for each of the initially saved
            # episodes after deduplication and resolution of edges is a complex task.
            # This current implementation focuses on bulk ingesting all unique nodes and edges.
            # A more advanced version might track precise edge linkages back to each source episode.

            # All extracted, processed, and deduplicated data is now passed to the provider's
            # bulk add method for efficient database persistence.
            await self.provider.add_nodes_and_edges_bulk(
                episodic_nodes=episodes, # The initial list of (now saved) episodes
                episodic_edges=episodic_edges_final, # Final list of episodic edges
                entity_nodes=nodes, # Final list of unique/deduplicated entity nodes
                entity_edges=entity_edges_final, # Final list of unique/deduplicated entity edges
                embedder=self.embedder # Embedder for any potential internal embedding needs by the provider
            )

            logger.info(f'Completed add_episode_bulk processing and delegated to provider for bulk saving.')
            end = time()
            logger.info(f'Total time for add_episode_bulk: {(end - start) * 1000} ms')

        except Exception as e:
            raise e

    async def build_communities(self, group_ids: list[str] | None = None) -> list[CommunityNode]:
        """
        Use a community clustering algorithm to find communities of nodes. Create community nodes summarising
        the content of these communities.
        ----------
        query : list[str] | None
            Optional. Create communities only for the listed group_ids. If blank the entire graph will be used.
        """
        # Clear existing communities
        await remove_communities(self.provider) # Pass provider

        community_nodes, community_edges = await build_communities(
            self.provider, self.llm_client, group_ids # Pass provider
        )

        await semaphore_gather(
            *[node.generate_name_embedding(self.embedder) for node in community_nodes]
        )

        # Save nodes and edges using provider methods
        await semaphore_gather(*[self.provider.save_community_node(node) for node in community_nodes])
        await semaphore_gather(*[self.provider.save_community_edge(edge) for edge in community_edges])

        return community_nodes

    async def search(
        self,
        query: str,
        center_node_uuid: str | None = None,
        group_ids: list[str] | None = None,
        num_results=DEFAULT_SEARCH_LIMIT,
        search_filter: SearchFilters | None = None,
    ) -> list[EntityEdge]:
        """
        Perform a hybrid search on the knowledge graph.

        This method executes a search query on the graph, combining vector and
        text-based search techniques to retrieve relevant facts, returning the edges as a string.

        This is our basic out-of-the-box search, for more robust results we recommend using our more advanced
        search method graphiti.search_().

        Parameters
        ----------
        query : str
            The search query string.
        center_node_uuid: str, optional
            Facts will be reranked based on proximity to this node
        group_ids : list[str | None] | None, optional
            The graph partitions to return data from.
        num_results : int, optional
            The maximum number of results to return. Defaults to 10.

        Returns
        -------
        list
            A list of EntityEdge objects that are relevant to the search query.

        Notes
        -----
        This method uses a SearchConfig with num_episodes set to 0 and
        num_results set to the provided num_results parameter.

        The search is performed using the current date and time as the reference
        point for temporal relevance.
        """
        search_config = (
            EDGE_HYBRID_SEARCH_RRF if center_node_uuid is None else EDGE_HYBRID_SEARCH_NODE_DISTANCE
        )
        search_config.limit = num_results

        edges = (
            await search(
                self.clients,
                query,
                group_ids,
                search_config,
                search_filter if search_filter is not None else SearchFilters(),
                center_node_uuid,
            )
        ).edges

        return edges

    async def _search(
        self,
        query: str,
        config: SearchConfig,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
    ) -> SearchResults:
        """DEPRECATED"""
        return await self.search_(
            query, config, group_ids, center_node_uuid, bfs_origin_node_uuids, search_filter
        )

    async def search_(
        self,
        query: str,
        config: SearchConfig = COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
    ) -> SearchResults:
        """search_ (replaces _search) is our advanced search method that returns Graph objects (nodes and edges) rather
        than a list of facts. This endpoint allows the end user to utilize more advanced features such as filters and
        different search and reranker methodologies across different layers in the graph.

        For different config recipes refer to search/search_config_recipes.
        """

        return await search(
            self.clients,
            query,
            group_ids,
            config,
            search_filter if search_filter is not None else SearchFilters(),
            center_node_uuid,
            bfs_origin_node_uuids,
        )

    async def get_nodes_and_edges_by_episode(self, episode_uuids: list[str]) -> SearchResults:
        episodes = await self.provider.get_episodic_nodes_by_uuids(uuids=episode_uuids)

        # Collect all entity_edge UUIDs from the episodes
        all_entity_edge_uuids = []
        for episode in episodes:
            if episode.entity_edges:
                all_entity_edge_uuids.extend(episode.entity_edges)

        # Fetch unique entity edges
        unique_entity_edge_uuids = list(dict.fromkeys(all_entity_edge_uuids))
        edges: list[EntityEdge] = []
        if unique_entity_edge_uuids:
            edges = await self.provider.get_entity_edges_by_uuids(uuids=unique_entity_edge_uuids)


        nodes = await self.provider.get_mentioned_nodes(episodes=episodes)

        return SearchResults(edges=edges, nodes=nodes, episodes=episodes, communities=[]) # Include episodes

    async def add_triplet(self, source_node: EntityNode, edge: EntityEdge, target_node: EntityNode):
        if source_node.name_embedding is None:
            await source_node.generate_name_embedding(self.embedder)
        if target_node.name_embedding is None:
            await target_node.generate_name_embedding(self.embedder)
        if edge.fact_embedding is None:
            await edge.generate_embedding(self.embedder)

        resolved_nodes, uuid_map = await resolve_extracted_nodes( # Uses self.clients
            self.clients,
            [source_node, target_node],
        )

        updated_edge = resolve_edge_pointers([edge], uuid_map)[0]

        # Use provider methods for get_relevant_edges and get_edge_invalidation_candidates
        related_edges_results = await self.provider.get_relevant_edges(edges=[updated_edge], search_filter=SearchFilters())
        related_edges = related_edges_results[0] if related_edges_results else []

        existing_edges_results = await self.provider.get_edge_invalidation_candidates(edges=[updated_edge], search_filter=SearchFilters())
        existing_edges = existing_edges_results[0] if existing_edges_results else []


        resolved_edge, invalidated_edges = await resolve_extracted_edge( # Uses llm_client
            self.llm_client,
            updated_edge,
            related_edges,
            existing_edges,
            EpisodicNode( # Dummy episode for context
                name='',
                source=EpisodeType.text,
                source_description='',
                content='',
                valid_at=edge.valid_at or utc_now(),
                entity_edges=[],
                group_id=edge.group_id,
            ),
        )

        # add_nodes_and_edges_bulk utility takes provider
        await add_nodes_and_edges_bulk(
            self.provider, [], [], resolved_nodes, [resolved_edge] + invalidated_edges, self.embedder
        )

    async def remove_episode(self, episode_uuid: str):
        # Find the episode to be deleted
        episode = await self.provider.get_episodic_node_by_uuid(uuid=episode_uuid)
        if not episode:
            logger.warning(f"Episode {episode_uuid} not found for deletion.")
            return

        # Find edges mentioned by the episode
        edges: List[EntityEdge] = []
        if episode.entity_edges:
            edges = await self.provider.get_entity_edges_by_uuids(uuids=episode.entity_edges)

        # We should only delete edges created by the episode
        edges_to_delete_uuids: list[str] = []
        for edge_item in edges: # Renamed to avoid conflict with outer 'edges'
            if edge_item.episodes and episode.uuid in edge_item.episodes: # Check if this episode is in the list
                if len(edge_item.episodes) == 1: # Only delete if this is the *only* episode mentioning it
                    edges_to_delete_uuids.append(edge_item.uuid)
                else:
                    # If mentioned by other episodes, just remove this episode's UUID from the edge's list
                    edge_item.episodes = [ep_uuid for ep_uuid in edge_item.episodes if ep_uuid != episode.uuid]
                    await self.provider.save_entity_edge(edge_item) # Re-save the edge with updated episodes list


        # Find nodes mentioned by the episode
        # This get_mentioned_nodes returns nodes linked via MENTIONS rel for Kuzu, or via episode.entity_edges for Neo4j's interpretation
        # The original logic implies nodes directly linked to this single episode.
        nodes_linked_to_episode = await self.provider.get_mentioned_nodes(episodes=[episode])

        nodes_to_delete_uuids: list[str] = []
        for node in nodes_linked_to_episode:
            # Use the new provider method to count mentions for the node.
            # If a node is only mentioned by the episode being deleted (i.e., mention count is 1),
            # then it's a candidate for deletion.
            mention_count = await self.provider.count_node_mentions(node_uuid=node.uuid)
            if mention_count == 1: # Only mentioned by this episode (or its associated edges)
                 # Further check: ensure this node is not part of edges NOT being deleted.
                 # This logic can get very complex. A simple check: if all edges connected to this node are in edges_to_delete_uuids.
                 # For now, the primary check is based on the direct mention count from episodes.
                 # If an entity node is only "mentioned" in the context of this one episode, we can delete it.
                nodes_to_delete_uuids.append(node.uuid)
            elif mention_count == 0: # Should not happen if get_mentioned_nodes returned it for this episode
                logger.warning(f"Node {node.uuid} returned by get_mentioned_nodes for episode {episode.uuid} but count_node_mentions is 0.")


        if edges_to_delete_uuids:
            await semaphore_gather(*[self.provider.delete_edge(edge_uuid) for edge_uuid in edges_to_delete_uuids])

        if nodes_to_delete_uuids:
            logger.info(f"Deleting nodes exclusively mentioned by episode {episode.uuid}: {nodes_to_delete_uuids}")
            await semaphore_gather(*[self.provider.delete_node(node_uuid) for node_uuid in nodes_to_delete_uuids])

        await self.provider.delete_node(episode.uuid) # Delete the episode itself
