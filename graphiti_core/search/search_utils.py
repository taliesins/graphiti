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
from collections import defaultdict
from time import time
from typing import Any

import numpy as np
# from neo4j import AsyncDriver, Query # AsyncDriver no longer used directly
from numpy._typing import NDArray
# from typing_extensions import LiteralString # LiteralString not used after refactor

from graphiti_core.edges import EntityEdge # get_entity_edge_from_record is provider specific now
# from graphiti_core.helpers import ( # DEFAULT_DATABASE, RUNTIME_QUERY, lucene_sanitize are provider specific
#     DEFAULT_DATABASE,
#     RUNTIME_QUERY,
#     lucene_sanitize,
#     normalize_l2,
#     semaphore_gather,
# )
from graphiti_core.helpers import normalize_l2, semaphore_gather # These are general helpers
from graphiti_core.nodes import ( # ENTITY_NODE_RETURN and get_..._from_record are provider specific
    # ENTITY_NODE_RETURN,
    CommunityNode,
    EntityNode,
    EpisodicNode,
    # get_community_node_from_record,
    # get_entity_node_from_record,
    # get_episodic_node_from_record,
)
from graphiti_core.providers.base import GraphDatabaseProvider # Import provider
from graphiti_core.search.search_filters import ( # Filter constructors are now provider specific
    SearchFilters,
    # edge_search_filter_query_constructor, # Provider specific
    # node_search_filter_query_constructor, # Provider specific
)

logger = logging.getLogger(__name__)

RELEVANT_SCHEMA_LIMIT = 10 # This can remain as a general constant
DEFAULT_MIN_SCORE = 0.6   # General constant
DEFAULT_MMR_LAMBDA = 0.5  # General constant
# MAX_SEARCH_DEPTH = 3      # Provider specific (used in Neo4jProvider queries)
# MAX_QUERY_LENGTH = 32     # Provider specific (used in Neo4jProvider for lucene)


# fulltext_query helper was Neo4j specific (lucene), moved to Neo4jProvider as _fulltext_lucene_query
# Search functions like edge_fulltext_search, node_similarity_search, etc., are now methods on the provider.
# This file will now primarily contain algorithmic helpers or high-level orchestration if needed,
# but most specific search logic belongs to providers.

# The following functions from the original search_utils.py are either:
# 1. Moved to providers (if they make DB calls, e.g., edge_fulltext_search, get_embeddings_for_nodes)
# 2. Kept here if purely algorithmic (e.g., rrf, maximal_marginal_relevance)
# 3. Refactored to use provider if they were helpers making DB calls (e.g., get_mentioned_nodes)

async def get_episodes_by_mentions(
    provider: GraphDatabaseProvider,
    edges: list[EntityEdge], # Removed unused 'nodes' parameter
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EpisodicNode]:
    """
    Retrieves episodic nodes associated with a list of entity edges.
    Note: This function seems to be unused by search.py and might be a candidate for removal.
    """
    episode_uuids: list[str] = []
    for edge in edges:
        if edge.episodes: # Ensure episodes list is not None
            episode_uuids.extend(edge.episodes)

    if not episode_uuids:
        return []

    # Use provider.get_episodic_nodes_by_uuids
    # Take unique UUIDs before querying.
    unique_episode_uuids = list(dict.fromkeys(episode_uuids)) # Preserve order while making unique

    # The limit here applies to how many UUIDs to fetch.
    episodes = await provider.get_episodic_nodes_by_uuids(uuids=unique_episode_uuids[:limit])
    return episodes


# get_mentioned_nodes, get_communities_by_nodes are now methods on the provider interface
# So, direct calls to them from search.py will use provider.get_mentioned_nodes(...)

# hybrid_node_search was an internal helper for search.py, its logic will be part of the main search function using provider methods.

# get_relevant_nodes, get_relevant_edges, get_edge_invalidation_candidates are now methods on the provider interface.


# Algorithmic helpers - these can stay as they don't do I/O
def rrf(results: list[list[str]], rank_const=1, min_score: float = 0) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for result in results:
        for i, uuid in enumerate(result):
            scores[uuid] += 1 / (i + rank_const)

    scored_uuids = [term for term in scores.items()]
    scored_uuids.sort(reverse=True, key=lambda term: term[1])
    sorted_uuids = [term[0] for term in scored_uuids]
    return [uuid for uuid in sorted_uuids if scores[uuid] >= min_score]


async def node_distance_reranker( # This function makes DB calls via driver
    provider: GraphDatabaseProvider,
    node_uuids: list[str],
    center_node_uuid: str,
    min_score: float = 0,
) -> list[str]:
    """
    Reranks a list of node UUIDs based on their shortest path distance to a center node.
    Nodes closer to the center_node_uuid are ranked higher.
    The Cypher query for shortest path may have provider-specific syntax or performance characteristics.
    """
    filtered_uuids = list(filter(lambda node_uuid: node_uuid != center_node_uuid, node_uuids))
    scores: dict[str, float] = {center_node_uuid: 0.0}

    # This query needs to be executed by the provider
    # The query itself might be Neo4j specific (SHORTEST)
    # TODO: Consider if this reranker should be part of the provider, or if the query needs to be generic
    # For now, assume provider.execute_query can run this if it's a Neo4jProvider
    query_str = """
        UNWIND $node_uuids AS node_uuid
        MATCH p = SHORTEST((center:Entity {uuid: $center_uuid})-[:RELATES_TO*]-(n:Entity {uuid: node_uuid}))
        RETURN length(p) AS score, node_uuid AS uuid
        """
        # Note: SHORTEST 1 used to be SHORTEST. Kuzu/other DBs might need different syntax.
        # Neo4j's `SHORTEST` implies `*` (any length), `SHORTEST 1` is not standard.
        # `[:RELATES_TO*]` is variable length. `[:RELATES_TO]-+` is also Neo4j specific.
        # A more standard variable length is `(center)-[:RELATES_TO*1..10]-(n)` for paths up to 10 hops.
        # For now, keeping a Neo4j-like query, assuming the provider can handle it or this util is Neo4j-specific.

    path_results, _, _ = await provider.execute_query( # Changed driver.execute_query
        query_str,
        params={"node_uuids": filtered_uuids, "center_uuid": center_node_uuid},
        # database_=DEFAULT_DATABASE, # Provider handles database selection
        # routing_='r', # Provider handles routing if applicable
    )

    for result in path_results: # path_results is already list of dicts
        uuid = result['uuid']
        score = result['score']
        scores[uuid] = float(score) if score is not None else float('inf')


    for uuid in filtered_uuids:
        if uuid not in scores:
            scores[uuid] = float('inf')

    filtered_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

    if center_node_uuid in node_uuids:
        # scores[center_node_uuid] = 0.1 # This was arbitrary
        # Ensure center node is first if it was part of the original list and we want it ranked high
        if center_node_uuid in filtered_uuids: # Should not happen due to filter
             filtered_uuids.remove(center_node_uuid)
        filtered_uuids.insert(0, center_node_uuid)


    # The min_score logic for distance seems inverted (1/score), higher distance = lower rank.
    # If score is distance, then we want nodes where distance is small.
    # Original: (1 / scores[uuid]) >= min_score. If min_score=0, always true for finite scores.
    # If min_score is e.g. 0.5, then 1/distance >= 0.5 -> distance <= 2.
    # This seems okay.
    final_uuids = [uuid for uuid in filtered_uuids if scores[uuid] != float('inf') and (1.0 / (scores[uuid] + 1e-9)) >= min_score] # add epsilon for score 0
    if center_node_uuid in node_uuids and center_node_uuid not in final_uuids and scores[center_node_uuid] == 0.0 and (1.0 / 1e-9 >=min_score): # Center node itself
        final_uuids.insert(0,center_node_uuid)


    return list(dict.fromkeys(final_uuids)) # Preserve order and unique


async def episode_mentions_reranker( # This function makes DB calls via driver
    provider: GraphDatabaseProvider,
    node_uuids: list[list[str]], # This is list of lists of uuids, typically from multiple search sources
    min_score: float = 0
) -> list[str]:
    """
    Reranks node UUIDs based on the count of distinct episodic mentions.
    Nodes mentioned in more episodes are ranked higher.
    The underlying Cypher query's performance for counting mentions might vary by provider.
    """
    sorted_uuids_flat = rrf(node_uuids) # Flatten and initial rank using RRF
    scores: dict[str, float] = {uuid: 0.0 for uuid in sorted_uuids_flat}


    # This query needs to be executed by the provider
    # TODO: Similar to node_distance_reranker, this query might be Neo4j specific.
    query_str = """
        UNWIND $node_uuids AS node_uuid 
        MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity {uuid: node_uuid})
        RETURN count(DISTINCT episode) AS score, n.uuid AS uuid 
        """
        # Using count(DISTINCT episode) to count unique episodes mentioning the node.

    results, _, _ = await provider.execute_query( # Changed driver.execute_query
        query_str,
        params={"node_uuids": sorted_uuids_flat},
        # database_=DEFAULT_DATABASE, # Provider handles
        # routing_='r', # Provider handles
    )

    for result in results: # result is a dict
        scores[result['uuid']] = float(result['score'])

    # Rerank based on mention counts (higher is better)
    # sorted_uuids_flat.sort(key=lambda cur_uuid: scores.get(cur_uuid, 0.0), reverse=True)
    
    # Filter by min_score before sorting by score
    # Then sort the filtered list
    final_sorted_uuids = [uuid for uuid in sorted_uuids_flat if scores.get(uuid, 0.0) >= min_score]
    final_sorted_uuids.sort(key=lambda cur_uuid: scores.get(cur_uuid, 0.0), reverse=True)


    return final_sorted_uuids


def maximal_marginal_relevance(
    query_vector: list[float],
    candidates: dict[str, list[float]], # uuid -> embedding
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    min_score: float = -2.0, # MMR scores can be negative
) -> list[str]:
    if not candidates:
        return []
    if not query_vector: # Should not happen if query is always embedded
        return list(candidates.keys())

    start = time()
    query_array = normalize_l2(np.array(query_vector)) # Normalize query vector
    
    candidate_uuids = list(candidates.keys())
    candidate_embeddings_normalized = np.array([normalize_l2(candidates[uuid]) for uuid in candidate_uuids])

    # Similarities with query
    query_similarities = np.dot(candidate_embeddings_normalized, query_array.T)

    # Inter-candidate similarities
    inter_candidate_similarity_matrix = np.dot(candidate_embeddings_normalized, candidate_embeddings_normalized.T)
    np.fill_diagonal(inter_candidate_similarity_matrix, -np.inf) # Ignore self-similarity for max

    selected_uuids: list[str] = []
    remaining_indices = list(range(len(candidate_uuids)))

    while len(selected_uuids) < len(candidate_uuids):
        if not remaining_indices:
            break
        
        best_mmr_score = -np.inf
        best_idx_in_remaining = -1

        for i, current_idx in enumerate(remaining_indices):
            relevance_to_query = query_similarities[current_idx]
            
            max_similarity_to_selected = 0.0
            if selected_uuids: # If some items are already selected
                # Similarities of current candidate to already selected ones
                sim_to_selected = inter_candidate_similarity_matrix[current_idx, [candidate_uuids.index(su) for su in selected_uuids]]
                if sim_to_selected.size > 0 : # Ensure there are similarities to selected items
                    max_similarity_to_selected = np.max(sim_to_selected)
            
            mmr_score = mmr_lambda * relevance_to_query - (1 - mmr_lambda) * max_similarity_to_selected
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx_in_remaining = i
        
        if best_idx_in_remaining == -1: # Should not happen if remaining_indices is not empty
            break 
            
        selected_uuid_idx = remaining_indices.pop(best_idx_in_remaining)
        selected_uuids.append(candidate_uuids[selected_uuid_idx])

    end = time()
    logger.debug(f'Completed MMR reranking in {(end - start) * 1000} ms, selected {len(selected_uuids)} items.')
    
    # min_score for MMR is tricky as scores are relative.
    # This simplistic filter might not be ideal for MMR's output.
    # For now, returning all selected_uuids as MMR itself is a ranking and selection process.
    # If min_score needs to apply to the relevance_to_query part, that's a different filter.
    return selected_uuids


# get_embeddings_for_nodes, _communities, _edges are now methods on the provider.
# The old versions in search_utils.py that took a driver should be removed or this file heavily refactored.
# For now, assuming they are removed and calls in search.py will go to provider.
