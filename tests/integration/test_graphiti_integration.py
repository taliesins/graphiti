import pytest
import pytest_asyncio
import os
import shutil
from typing import List
from datetime import datetime, timezone, timedelta

from graphiti_core.graphiti import Graphiti, AddEpisodeResults
from graphiti_core.providers.neo4j_provider import Neo4jProvider
from graphiti_core.providers.kuzudb_provider import KuzuDBProvider
from graphiti_core.providers.base import GraphDatabaseProvider
from graphiti_core.llm_client import LLMClient # For mocking
from graphiti_core.embedder import EmbedderClient # For mocking
from graphiti_core.cross_encoder.client import CrossEncoderClient # For mocking
from graphiti_core.nodes import EpisodeType, EntityNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_config import SearchConfig, SearchType, RankerType

# --- Environment Configuration (Ideally from .env or test config) ---
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j") # Default graphiti database

KUZU_DB_PATH_INT_TEST = "./kuzu_int_test_db" # Use a dedicated path for integration test DB

# --- Mock Clients ---
@pytest.fixture(scope="module")
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    # Define mock behaviors if needed for specific tests, e.g., LLM-based summarization
    async def mock_generate_response(*args, **kwargs):
        # Based on prompt, return some mock data
        prompt_name = kwargs.get("prompt_name", "") # Assuming prompt_name is passed
        if "summarize_nodes.summarize_pair" in str(args[0]): # Fragile check
            return {"summary": "Mocked combined summary"}
        if "summarize_nodes.summary_description" in str(args[0]):
            return {"description": "Mocked community name"}
        # Add other mock responses as needed for community building, etc.
        return {}
    client.generate_response = AsyncMock(side_effect=mock_generate_response)
    return client

@pytest.fixture(scope="module")
def mock_embedder_client():
    client = MagicMock(spec=EmbedderClient)
    # Simple mock that returns a fixed-size vector of zeros
    async def mock_create_batch(texts: List[str]) -> List[List[float]]:
        return [[0.0] * 128 for _ in texts] # Assuming embedding dim 128 for test
    async def mock_create(input_data: List[str]) -> List[List[float]]: # Kuzu provider uses this
         return [[0.0] * 1536] # Match default Kuzu embedding dim
    client.create_batch = AsyncMock(side_effect=mock_create_batch)
    client.create = AsyncMock(side_effect=mock_create)
    return client

@pytest.fixture(scope="module")
def mock_cross_encoder_client():
    client = MagicMock(spec=CrossEncoderClient)
    async def mock_rerank(query: str, documents: List[str], limit: int) -> List[int]:
        # Simple pass-through reranking (indices [0, 1, ..., limit-1])
        return list(range(min(limit, len(documents))))
    client.rerank = AsyncMock(side_effect=mock_rerank)
    return client


# --- Provider Fixtures ---
@pytest_asyncio.fixture(scope="function")
async def neo4j_provider_instance():
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        pytest.skip("Neo4j environment variables not set, skipping integration tests.")

    provider = Neo4jProvider(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)
    await provider.connect()
    await provider.clear_data() # Clear before test
    await provider.build_indices_and_constraints(delete_existing=True) # Ensure schema is fresh
    yield provider
    # await provider.clear_data() # Optional: Clear after test
    await provider.close()

@pytest_asyncio.fixture(scope="function")
async def kuzu_provider_instance():
    # Ensure clean state for Kuzu by removing the DB directory if it exists
    if os.path.exists(KUZU_DB_PATH_INT_TEST):
        shutil.rmtree(KUZU_DB_PATH_INT_TEST)
    os.makedirs(KUZU_DB_PATH_INT_TEST, exist_ok=True)

    provider = KuzuDBProvider(database_path=KUZU_DB_PATH_INT_TEST, in_memory=False) # Use on-disk for persistence if needed across calls
    await provider.connect()
    # KuzuDBProvider.clear_data calls build_indices_and_constraints(delete_existing=True)
    await provider.clear_data() # This will also build schema
    yield provider
    await provider.close()
    # Clean up KuzuDB directory after test run
    if os.path.exists(KUZU_DB_PATH_INT_TEST):
        shutil.rmtree(KUZU_DB_PATH_INT_TEST)

# --- Graphiti Fixtures ---
@pytest.fixture
def graphiti_neo4j(neo4j_provider_instance, mock_llm_client, mock_embedder_client, mock_cross_encoder_client):
    return Graphiti(
        provider=neo4j_provider_instance,
        llm_client=mock_llm_client,
        embedder=mock_embedder_client,
        cross_encoder=mock_cross_encoder_client
    )

@pytest.fixture
def graphiti_kuzu(kuzu_provider_instance, mock_llm_client, mock_embedder_client, mock_cross_encoder_client):
    # Adjust mock_embedder_client if Kuzu expects different embedding dim for some tests
    # For now, using the same one. Kuzu provider has default dim 1536.
    # Mock embedder's create method for Kuzu's default dim if needed:
    async def mock_create_kuzu(input_data: List[str]) -> List[List[float]]:
         return [[0.0] * 1536]
    mock_embedder_client.create = AsyncMock(side_effect=mock_create_kuzu)

    return Graphiti(
        provider=kuzu_provider_instance,
        llm_client=mock_llm_client,
        embedder=mock_embedder_client,
        cross_encoder=mock_cross_encoder_client
    )

# --- Parametrize tests for both providers ---
graphiti_instances = ["graphiti_neo4j", "graphiti_kuzu"]

@pytest.mark.asyncio
@pytest.mark.parametrize("graphiti_fixture_name", graphiti_instances)
async def test_add_episode_and_verify_data(graphiti_fixture_name, request):
    graphiti: Graphiti = request.getfixturevalue(graphiti_fixture_name)
    provider = graphiti.provider # Get the actual provider instance for direct checks

    episode_name = "Test Episode Alpha"
    episode_body = "Entity Alice talks to Entity Bob about Project X."
    source_desc = "Test source"
    ref_time = datetime.now(timezone.utc)
    group_id = "group_alpha"

    # Mock LLM responses for node/edge extraction if not already broadly mocked
    # For this test, we assume the LLM client fixture handles it or extraction is simple enough.
    # If specific entities/edges are expected, LLM mock needs to return them.
    # For now, we'll check for creation of *some* nodes/edges.

    add_results: AddEpisodeResults = await graphiti.add_episode(
        name=episode_name,
        episode_body=episode_body,
        source_description=source_desc,
        reference_time=ref_time,
        group_id=group_id,
        source=EpisodeType.text
    )

    assert add_results is not None
    assert add_results.episode is not None
    assert add_results.episode.name == episode_name
    assert add_results.episode.group_id == group_id

    # Verify EpisodicNode in DB
    retrieved_episode = await provider.get_episodic_node_by_uuid(add_results.episode.uuid)
    assert retrieved_episode is not None
    assert retrieved_episode.name == episode_name

    # Verify EntityNodes and Edges (existence and basic properties)
    # The exact nodes/edges created depend on LLM extraction logic.
    # For a robust integration test, mock LLM to return predictable entities/edges.
    # Here, we check if *any* nodes/edges were created as listed in AddEpisodeResults.

    assert len(add_results.nodes) > 0, "Expected some entity nodes to be created"
    assert len(add_results.edges) > 0, "Expected some entity edges to be created"

    for entity_node in add_results.nodes:
        retrieved_entity = await provider.get_entity_node_by_uuid(entity_node.uuid)
        assert retrieved_entity is not None
        assert retrieved_entity.name == entity_node.name
        assert retrieved_entity.group_id == group_id
        # Check for MENTIONS relationship to the episode (via EpisodicEdge)
        # This requires checking EpisodicEdges or querying relationships.
        # For simplicity, we assume add_episode correctly creates these if nodes/edges are formed.

    # Verify one edge
    test_edge_uuid = add_results.edges[0].uuid
    retrieved_edge = await provider.get_entity_edge_by_uuid(test_edge_uuid)
    assert retrieved_edge is not None
    assert retrieved_edge.group_id == group_id
    assert add_results.episode.uuid in retrieved_edge.episodes

    # Verify search finds something related
    search_results_nodes = await graphiti.search_(query="Alice", group_ids=[group_id], config=SearchConfig(search_type=SearchType.NODE_ONLY))
    search_results_edges = await graphiti.search_(query="Project X", group_ids=[group_id], config=SearchConfig(search_type=SearchType.EDGE_ONLY))

    # These assertions depend heavily on mocked LLM output for node/edge extraction.
    # If LLM is not mocked to extract "Alice" or "Project X", these might fail or be empty.
    # For a basic "something was indexed" check:
    if "Alice" in episode_body: # Crude check
         assert len(search_results_nodes.nodes) > 0 or len(search_results_nodes.edges) > 0 # Search might return edges if node name in edge fact

    if "Project X" in episode_body:
        assert len(search_results_edges.edges) > 0 or len(search_results_edges.nodes) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("graphiti_fixture_name", graphiti_instances)
async def test_remove_episode(graphiti_fixture_name, request):
    graphiti: Graphiti = request.getfixturevalue(graphiti_fixture_name)
    provider = graphiti.provider

    episode_name = "Episode to Delete"
    episode_body = "Entity Charlie creates Document Y."
    source_desc = "Removable source"
    ref_time = datetime.now(timezone.utc)
    group_id = "group_removable"

    # Add an episode
    add_results = await graphiti.add_episode(
        name=episode_name, episode_body=episode_body, source_description=source_desc,
        reference_time=ref_time, group_id=group_id, source=EpisodeType.text
    )
    episode_uuid_to_delete = add_results.episode.uuid
    created_nodes_uuids = [n.uuid for n in add_results.nodes]
    created_edges_uuids = [e.uuid for e in add_results.edges]

    assert await provider.get_episodic_node_by_uuid(episode_uuid_to_delete) is not None
    if created_nodes_uuids:
        assert await provider.get_entity_node_by_uuid(created_nodes_uuids[0]) is not None
    if created_edges_uuids:
        assert await provider.get_entity_edge_by_uuid(created_edges_uuids[0]) is not None

    # Remove the episode
    await graphiti.remove_episode(episode_uuid_to_delete)

    # Verify episode is deleted
    assert await provider.get_episodic_node_by_uuid(episode_uuid_to_delete) is None

    # Verify that nodes mentioned ONLY in this episode are deleted.
    # This depends on count_node_mentions and the extraction logic.
    # If "Charlie" and "Document Y" were extracted:
    for node_uuid in created_nodes_uuids:
        # If node was only mentioned in this episode (mention count became 0 after this episode's edges removed, then 1 before)
        # it should be deleted.
        assert await provider.get_entity_node_by_uuid(node_uuid) is None, f"Node {node_uuid} should have been deleted"

    # Verify that edges created ONLY by this episode are deleted.
    for edge_uuid in created_edges_uuids:
        assert await provider.get_entity_edge_by_uuid(edge_uuid) is None, f"Edge {edge_uuid} should have been deleted"


@pytest.mark.asyncio
@pytest.mark.parametrize("graphiti_fixture_name", graphiti_instances)
async def test_add_episode_bulk_and_verify_data(graphiti_fixture_name, request):
    graphiti: Graphiti = request.getfixturevalue(graphiti_fixture_name)
    provider = graphiti.provider

    now = datetime.now(timezone.utc)
    raw_episodes = [
        RawEpisode(name="Bulk Ep 1", content="Entity X meets Entity Y.", source_description="Bulk source 1", source=EpisodeType.text, reference_time=now - timedelta(minutes=10)),
        RawEpisode(name="Bulk Ep 2", content="Entity Y discusses Project Z with Entity X.", source_description="Bulk source 2", source=EpisodeType.text, reference_time=now - timedelta(minutes=5)),
    ]
    group_id = "group_bulk"

    # Mock LLM responses for node/edge extraction to ensure predictable entities
    # This is crucial for verifying bulk processing results.
    # For this example, we'll assume the LLM mock is general or we check for general data creation.
    # A more robust test would mock LLM to extract X, Y, Z specifically.

    await graphiti.add_episode_bulk(bulk_episodes=raw_episodes, group_id=group_id)

    # Verify EpisodicNodes
    episodes_in_db = await provider.get_episodic_nodes_by_group_ids(group_ids=[group_id], limit=10)
    assert len(episodes_in_db) == 2
    ep_names_in_db = {ep.name for ep in episodes_in_db}
    assert "Bulk Ep 1" in ep_names_in_db
    assert "Bulk Ep 2" in ep_names_in_db

    # Verify EntityNodes (highly dependent on LLM mock)
    # We expect "Entity X", "Entity Y", "Project Z" if LLM works as expected.
    # For now, let's check if some entities were created in the group.
    entity_nodes_in_db = await provider.get_entity_nodes_by_group_ids(group_ids=[group_id], limit=10)
    assert len(entity_nodes_in_db) > 0 # At least one entity should be created.

    # Verify EntityEdges (also LLM dependent)
    entity_edges_in_db = await provider.get_entity_edges_by_group_ids(group_ids=[group_id], limit=10)
    assert len(entity_edges_in_db) > 0 # At least one edge.


    # Verify search can find some of this data
    # This again depends on LLM mock and extraction.
    search_results = await graphiti.search_(
        query="Entity X",
        group_ids=[group_id],
        config=SearchConfig(search_type=SearchType.NODE_ONLY, limit=5)
    )
    if "Entity X" in raw_episodes[0].content + raw_episodes[1].content:
        assert len(search_results.nodes) > 0 or len(search_results.edges) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("graphiti_fixture_name", graphiti_instances)
async def test_search_advanced(graphiti_fixture_name, request):
    graphiti: Graphiti = request.getfixturevalue(graphiti_fixture_name)
    provider = graphiti.provider
    group_id = "search_group"

    # Add some data
    await graphiti.add_episode(
        name="Search Test Ep",
        episode_body="Alpha is friends with Beta. Gamma works on Project Delta with Alpha.",
        source_description="Search test data",
        reference_time=datetime.now(timezone.utc),
        group_id=group_id,
        source=EpisodeType.text
    )

    # Assuming LLM extracts "Alpha", "Beta", "Gamma", "Project Delta" and relevant edges.
    # Perform a search
    search_config = SearchConfig(
        search_type=SearchType.COMBINED, # Default, gets nodes and edges
        limit=5,
        node_search_config=SearchConfig(search_type=SearchType.NODE_HYBRID, ranker=RankerType.RRF, limit=3),
        edge_search_config=SearchConfig(search_type=SearchType.EDGE_HYBRID, ranker=RankerType.RRF, limit=3)
    )

    search_results = await graphiti.search_(
        query="Alpha Project Delta",
        group_ids=[group_id],
        config=search_config
    )

    assert search_results is not None
    # We expect some nodes and/or edges. Exact count depends on extraction and ranking.
    assert len(search_results.nodes) > 0 or len(search_results.edges) > 0

    found_alpha = any(node.name.lower() == "alpha" for node in search_results.nodes)
    # Project Delta might be an entity or part of an edge fact
    found_delta_related = found_alpha or \
                          any("delta" in edge.fact.lower() for edge in search_results.edges) or \
                          any(node.name.lower() == "project delta" for node in search_results.nodes)

    assert found_delta_related, "Expected search results related to 'Alpha' or 'Project Delta'"


# Neo4j specific test for now, as community building logic might be provider-specific
@pytest.mark.asyncio
async def test_build_communities_neo4j(graphiti_neo4j: Graphiti, mock_llm_client: MagicMock):
    graphiti = graphiti_neo4j # Use the specific fixture
    provider = graphiti.provider
    group_id = "community_group"

    # Add interconnected data
    # Node A -- relates to --> Node B
    # Node B -- relates to --> Node C
    # Node A -- relates to --> Node C
    # Node D -- relates to --> Node E
    # Expect two communities: (A,B,C) and (D,E)

    # Mock LLM to extract these specific nodes and edges
    # For simplicity, we'll manually create them via provider for this test
    # to bypass complex LLM mocking for node/edge extraction.

    nodes_to_create = [
        EntityNode(uuid="nA", name="Node A", group_id=group_id, summary="A related to B and C"),
        EntityNode(uuid="nB", name="Node B", group_id=group_id, summary="B related to A and C"),
        EntityNode(uuid="nC", name="Node C", group_id=group_id, summary="C related to A and B"),
        EntityNode(uuid="nD", name="Node D", group_id=group_id, summary="D related to E"),
        EntityNode(uuid="nE", name="Node E", group_id=group_id, summary="E related to D"),
    ]
    for node in nodes_to_create:
        await provider.save_entity_node(node)

    edges_to_create = [
        EntityEdge(source_node_uuid="nA", target_node_uuid="nB", name="RELATES_TO", fact="A-B", group_id=group_id, created_at=datetime.now(timezone.utc)),
        EntityEdge(source_node_uuid="nB", target_node_uuid="nC", name="RELATES_TO", fact="B-C", group_id=group_id, created_at=datetime.now(timezone.utc)),
        EntityEdge(source_node_uuid="nA", target_node_uuid="nC", name="RELATES_TO", fact="A-C", group_id=group_id, created_at=datetime.now(timezone.utc)),
        EntityEdge(source_node_uuid="nD", target_node_uuid="nE", name="RELATES_TO", fact="D-E", group_id=group_id, created_at=datetime.now(timezone.utc)),
    ]
    for edge in edges_to_create:
        await provider.save_entity_edge(edge)

    # Mock LLM for community summarization (already done by mock_llm_client fixture)
    # mock_llm_client.generate_response.side_effect = ... (if more specific responses are needed)

    # Run build_communities
    community_nodes = await graphiti.build_communities(group_ids=[group_id])

    assert community_nodes is not None
    # Exact number of communities depends on the algorithm (label propagation in Neo4j provider)
    # and LLM summarization. For this structure, expect 2.
    assert len(community_nodes) > 0 # At least one community

    # Verify HAS_MEMBER edges
    # Example: check if Node A is in a community
    # This requires a way to get edges for a community node, or query all HAS_MEMBER edges
    all_comm_edges = await provider.get_community_edges_by_group_ids(group_ids=[group_id], limit=100)

    assert len(all_comm_edges) > 0 # Some HAS_MEMBER edges should exist

    # Check if specific nodes are members of found communities
    node_a_community_found = False
    for comm_edge in all_comm_edges:
        # Assuming community node is source, entity node is target for HAS_MEMBER
        if comm_edge.target_node_uuid == "nA":
            node_a_community_found = True
            # Optionally, check properties of community_nodes[comm_edge.source_node_uuid]
            break
    assert node_a_community_found, "Node A should be part of a community"

```
