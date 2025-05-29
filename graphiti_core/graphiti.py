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
from graphiti_core.search.search_utils import (
    RELEVANT_SCHEMA_LIMIT,
    get_edge_invalidation_candidates,
    get_mentioned_nodes,
    get_relevant_edges,
)
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
                # EpisodicNode.get_by_uuids will be refactored to use provider
                else await EpisodicNode.get_by_uuids(self.provider, previous_episode_uuids) 
            )

            episode = (
                # EpisodicNode.get_by_uuid will be refactored to use provider
                await EpisodicNode.get_by_uuid(self.provider, uuid) 
                if uuid is not None
                else EpisodicNode(
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
                        update_community(self.driver, self.llm_client, self.embedder, node)
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
            # retrieve_previous_episodes_bulk will need refactoring to use provider
            episode_pairs = await retrieve_previous_episodes_bulk(self.provider, episodes) # type: ignore

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
            (nodes, uuid_map), extracted_edges_timestamped = await semaphore_gather(
                dedupe_nodes_bulk(self.driver, self.llm_client, extracted_nodes),
                extract_edge_dates_bulk(self.llm_client, extracted_edges, episode_pairs),
            )

            # Node.save will be refactored to use provider
            await semaphore_gather(*[node.save(self.provider) for node in nodes])

            # re-map edge pointers so that they don't point to discard dupe nodes
            extracted_edges_with_resolved_pointers: list[EntityEdge] = resolve_edge_pointers(
                extracted_edges_timestamped, uuid_map
            )
            episodic_edges_with_resolved_pointers: list[EpisodicEdge] = resolve_edge_pointers(
                episodic_edges, uuid_map
            )

            # Edge.save will be refactored to use provider
            await semaphore_gather(
                *[edge.save(self.provider) for edge in episodic_edges_with_resolved_pointers]
            )

            # Dedupe extracted edges
            # dedupe_edges_bulk will be refactored to use provider
            edges = await dedupe_edges_bulk(
                self.provider, self.llm_client, extracted_edges_with_resolved_pointers # type: ignore
            )
            logger.debug(f'extracted edge length: {len(edges)}')

            # invalidate edges

            # Edge.save will be refactored to use provider
            await semaphore_gather(*[edge.save(self.provider) for edge in edges])

            # The main bulk save operation will be done by the provider itself if we refactor utils
            # For now, individual saves are being delegated.
            # If add_nodes_and_edges_bulk utility is kept, it needs to use the provider.
            # Or, we directly call self.provider.add_nodes_and_edges_bulk here.
            # The current structure of add_episode_bulk is very utility-heavy.
            # Awaiting full refactor of those utils.

            end = time()
            logger.info(f'Completed add_episode_bulk in {(end - start) * 1000} ms')

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
        await remove_communities(self.driver)

        community_nodes, community_edges = await build_communities(
            self.driver, self.llm_client, group_ids
        )

        await semaphore_gather(
            *[node.generate_name_embedding(self.embedder) for node in community_nodes]
        )

        await semaphore_gather(*[node.save(self.driver) for node in community_nodes])
        await semaphore_gather(*[edge.save(self.driver) for edge in community_edges])

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
        episodes = await EpisodicNode.get_by_uuids(self.driver, episode_uuids)

        edges_list = await semaphore_gather(
            *[EntityEdge.get_by_uuids(self.driver, episode.entity_edges) for episode in episodes]
        )

        edges: list[EntityEdge] = [edge for lst in edges_list for edge in lst]

        nodes = await get_mentioned_nodes(self.driver, episodes)

        return SearchResults(edges=edges, nodes=nodes, episodes=[], communities=[])

    async def add_triplet(self, source_node: EntityNode, edge: EntityEdge, target_node: EntityNode):
        if source_node.name_embedding is None:
            await source_node.generate_name_embedding(self.embedder)
        if target_node.name_embedding is None:
            await target_node.generate_name_embedding(self.embedder)
        if edge.fact_embedding is None:
            await edge.generate_embedding(self.embedder)

        resolved_nodes, uuid_map = await resolve_extracted_nodes(
            self.clients,
            [source_node, target_node],
        )

        updated_edge = resolve_edge_pointers([edge], uuid_map)[0]

        related_edges = (await get_relevant_edges(self.driver, [updated_edge], SearchFilters()))[0]
        existing_edges = (
            await get_edge_invalidation_candidates(self.driver, [updated_edge], SearchFilters())
        )[0]

        resolved_edge, invalidated_edges = await resolve_extracted_edge(
            self.llm_client,
            updated_edge,
            related_edges,
            existing_edges,
            EpisodicNode(
                name='',
                source=EpisodeType.text,
                source_description='',
                content='',
                valid_at=edge.valid_at or utc_now(),
                entity_edges=[],
                group_id=edge.group_id,
            ),
        )

        await add_nodes_and_edges_bulk(
            self.driver, [], [], resolved_nodes, [resolved_edge] + invalidated_edges, self.embedder
        )

    async def remove_episode(self, episode_uuid: str):
        # Find the episode to be deleted
        episode = await EpisodicNode.get_by_uuid(self.driver, episode_uuid)

        # Find edges mentioned by the episode
        edges = await EntityEdge.get_by_uuids(self.driver, episode.entity_edges)

        # We should only delete edges created by the episode
        edges_to_delete: list[EntityEdge] = []
        for edge in edges:
            if edge.episodes and edge.episodes[0] == episode.uuid:
                edges_to_delete.append(edge)

        # Find nodes mentioned by the episode
        nodes = await get_mentioned_nodes(self.driver, [episode])
        # We should delete all nodes that are only mentioned in the deleted episode
        nodes_to_delete: list[EntityNode] = []
        for node in nodes:
            query: LiteralString = 'MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $uuid}) RETURN count(*) AS episode_count'
            records, _, _ = await self.driver.execute_query(
                query, uuid=node.uuid, database_=DEFAULT_DATABASE, routing_='r'
            )

            for record in records:
                if record['episode_count'] == 1:
                    nodes_to_delete.append(node)

        await semaphore_gather(*[node.delete(self.driver) for node in nodes_to_delete])
        await semaphore_gather(*[edge.delete(self.driver) for edge in edges_to_delete])
        await episode.delete(self.driver)
