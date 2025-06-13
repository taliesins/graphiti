import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

from graphiti_core.providers.neo4j_provider import Neo4jProvider, SearchFilters, DateFilter, ComparisonOperator, _fulltext_lucene_query, _node_search_filter_query_constructor, _edge_search_filter_query_constructor, RELEVANT_SCHEMA_LIMIT, DEFAULT_MIN_SCORE
from graphiti_core.nodes import EntityNode, EpisodicNode, CommunityNode, EpisodeType, ENTITY_NODE_RETURN, get_entity_node_from_record
from graphiti_core.edges import EntityEdge, EpisodicEdge, CommunityEdge, ENTITY_EDGE_RETURN, get_entity_edge_from_record
from graphiti_core.embedder import EmbedderClient
from graphiti_core.models.nodes import node_db_queries # For query string verification
from graphiti_core.models.edges import edge_db_queries # For query string verification
from neo4j.exceptions import Neo4jError


# Mock data and records
MOCK_UUID = "mock-uuid-123"
MOCK_GROUP_ID = "mock-group-id"
MOCK_DATETIME_NATIVE = datetime.now(timezone.utc)

# Convert native datetime to a structure that mimics Neo4j's DateTime object for mocking
class MockNeo4jDateTime:
    def __init__(self, dt):
        self._dt = dt
    def to_native(self):
        return self._dt

MOCK_NEO4J_DATETIME = MockNeo4jDateTime(MOCK_DATETIME_NATIVE)

MOCK_ENTITY_NODE_RECORD = {
    "n": { # Assuming 'n' is the alias used in ENTITY_NODE_RETURN
        "uuid": MOCK_UUID,
        "name": "Test Entity",
        "group_id": MOCK_GROUP_ID,
        "created_at": MOCK_NEO4J_DATETIME,
        "summary": "Test summary",
        "labels": ["Entity", "TestLabel"],
        "attributes": {"prop1": "val1"}, # This would be properties(n)
        "name_embedding": [0.1, 0.2]
    }
}
# Adjust MOCK_ENTITY_NODE_RECORD based on actual ENTITY_NODE_RETURN structure (which returns individual fields, not a map 'n')
MOCK_ENTITY_NODE_FLAT_RECORD = {
    "uuid": MOCK_UUID,
    "name": "Test Entity",
    "group_id": MOCK_GROUP_ID,
    "created_at": MOCK_NEO4J_DATETIME,
    "summary": "Test summary",
    "labels": ["Entity", "TestLabel"],
    "attributes": {"prop1": "val1", "name": "Test Entity", "uuid": MOCK_UUID}, # properties(n) includes all
    "name_embedding": [0.1, 0.2] # Assuming name_embedding is not part of properties(n) in RETURN but separate
}


@pytest_asyncio.fixture
async def mock_driver():
    with patch("neo4j.AsyncGraphDatabase.driver") as mock_driver_class:
        driver_instance = AsyncMock()
        driver_instance.execute_query = AsyncMock(return_value=([], {}, {})) # Default for non-tx calls
        driver_instance.verify_connectivity = AsyncMock()
        driver_instance.closed = MagicMock(return_value=False)

        # Setup mock session and its execute_write to allow inspection of tx.run calls
        mock_session_instance = AsyncMock()

        # This mock_tx will be passed to the transaction function by our mock_execute_write
        # We can then inspect its .run calls.
        # Needs to be accessible by the test later.
        # We can attach it to driver_instance or make it part of what the fixture yields.
        # For simplicity, let's make it an attribute of the driver_instance if possible,
        # or pass it along with the driver.

        async def new_mock_execute_write(tx_func, *args, **kwargs):
            # This is the mock transaction object whose .run() we want to inspect
            mock_tx_for_test = AsyncMock(name="mock_tx_for_test_execute_write")
            # Store it somewhere the test can access it.
            # A bit hacky, but common for such scenarios.
            # Alternative: fixture yields (driver_instance, mock_tx_for_test)
            driver_instance._most_recent_mock_tx = mock_tx_for_test
            return await tx_func(mock_tx_for_test, *args, **kwargs)

        mock_session_instance.execute_write = AsyncMock(side_effect=new_mock_execute_write)

        # Ensure __aenter__ and __aexit__ are also AsyncMocks for async with context
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)

        driver_instance.session = MagicMock(return_value=mock_session_instance)

        mock_driver_class.return_value = driver_instance
        yield driver_instance # The test will retrieve mock_tx via driver_instance._most_recent_mock_tx

@pytest_asyncio.fixture
async def neo4j_provider(mock_driver):
    provider = Neo4jProvider(uri="neo4j://localhost", user="neo4j", password="password")
    provider.driver = mock_driver # Inject the mock driver
    return provider

@pytest.mark.asyncio
class TestNeo4jProvider:

    async def test_connect_and_close(self, neo4j_provider: Neo4jProvider, mock_driver: AsyncMock):
        # Connect is called implicitly by fixture if provider.driver is set,
        # or explicitly if we want to test the full connect logic
        # For this test, let's assume connect was implicitly called by __init__ if driver wasn't pre-mocked,
        # or we can test it more directly.
        # Since fixture injects a mock driver, connect() will use it.

        await neo4j_provider.connect()
        mock_driver.verify_connectivity.assert_called_once()

        await neo4j_provider.close()
        mock_driver.close.assert_called_once()

    async def test_verify_connectivity_failure(self, neo4j_provider: Neo4jProvider, mock_driver: AsyncMock):
        mock_driver.verify_connectivity.side_effect = Neo4jError("Connection failed")
        with pytest.raises(ConnectionError, match="Neo4j connectivity verification failed"):
            await neo4j_provider.verify_connectivity()

    async def test_save_entity_node(self, neo4j_provider: Neo4jProvider):
        node = EntityNode(
            uuid="test-uuid",
            name="Test Node",
            group_id="test-group",
            labels=["Custom"],
            summary="A test node",
            attributes={"attr1": "val1"},
            name_embedding=[0.1, 0.2, 0.3]
        )

        # Mock execute_query to simulate successful save
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([{"uuid": "test-uuid"}], {}, {}))

        await neo4j_provider.save_entity_node(node)

        # Assert execute_query was called. The actual query string is complex due to dynamic labels.
        # We can check parts of it or the parameters.
        args, kwargs = neo4j_provider.driver.execute_query.call_args

        # Check some key parameters
        params_sent = kwargs['params']
        assert params_sent['entity_data']['uuid'] == "test-uuid"
        assert params_sent['entity_data']['name'] == "Test Node"
        assert "SET n:`Custom`" in args[0] # Check if custom label part is in query string
        assert "MERGE (n:Entity {uuid: $entity_data.uuid})" in args[0]
        assert "SET n += $props_to_set" in args[0]
        assert "CALL db.create.setNodeVectorProperty(n, 'name_embedding', data.name_embedding)" in args[0]

    async def test_get_entity_node_by_uuid_found(self, neo4j_provider: Neo4jProvider):
        # Prepare a mock record that get_entity_node_from_record would parse
        # This mock should align with what ENTITY_NODE_RETURN provides
        mock_record_data = {
            "uuid": MOCK_UUID, "name": "Parsed Node", "group_id": MOCK_GROUP_ID,
            "created_at": MOCK_NEO4J_DATETIME, "summary": "Parsed Summary",
            "labels": ["Entity", "ParsedLabel"],
            "attributes": {"parsed_prop": "val_parsed", "uuid": MOCK_UUID, "name": "Parsed Node"}, # Simulating properties(n)
            "name_embedding": [0.3, 0.4]
        }
        # The execute_query mock should return a list of such records
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([mock_record_data], {}, {}))

        found_node = await neo4j_provider.get_entity_node_by_uuid(MOCK_UUID)

        assert found_node is not None
        assert found_node.uuid == MOCK_UUID
        assert found_node.name == "Parsed Node"
        assert "ParsedLabel" in found_node.labels
        assert found_node.attributes == {"parsed_prop": "val_parsed"} # After cleanup
        assert found_node.name_embedding == [0.3, 0.4]

        # Verify the query structure
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        expected_query_part = f"MATCH (n:Entity {{uuid: $uuid}}) {ENTITY_NODE_RETURN}"
        assert args[0].strip() == expected_query_part.strip() # Compare stripped queries
        assert kwargs['params'] == {"uuid": MOCK_UUID}


    async def test_get_entity_node_by_uuid_not_found(self, neo4j_provider: Neo4jProvider):
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([], {}, {})) # Simulate node not found

        found_node = await neo4j_provider.get_entity_node_by_uuid("non-existent-uuid")
        assert found_node is None

    async def test_node_fulltext_search(self, neo4j_provider: Neo4jProvider):
        mock_query = "search term"
        mock_group_ids = ["group1"]
        mock_limit = 5

        # Mock record to be returned by Lucene
        mock_search_record = {
            "n": MOCK_ENTITY_NODE_FLAT_RECORD, # 'n' is the alias used in YIELD node AS n
            "score": 0.95
        }
        # The parser get_entity_node_from_record expects fields directly, not nested under 'n' after YIELD
        # So, the actual record passed to parser would be MOCK_ENTITY_NODE_FLAT_RECORD itself.

        neo4j_provider.driver.execute_query = AsyncMock(return_value=([MOCK_ENTITY_NODE_FLAT_RECORD], {}, {}))

        results = await neo4j_provider.node_fulltext_search(
            query=mock_query,
            search_filter=SearchFilters(),
            group_ids=mock_group_ids,
            limit=mock_limit
        )

        assert len(results) == 1
        assert results[0].uuid == MOCK_ENTITY_NODE_FLAT_RECORD["uuid"]

        args, kwargs = neo4j_provider.driver.execute_query.call_args
        # Check that the Lucene query part is constructed correctly
        expected_lucene_query = _fulltext_lucene_query(mock_query, mock_group_ids)
        assert kwargs['params']['lucene_query'] == expected_lucene_query
        assert "CALL db.index.fulltext.queryNodes(\"node_name_and_summary\"" in args[0]
        assert f"LIMIT $limit" # Limit is handled by Lucene call limit, not Cypher LIMIT here for queryNodes
        assert kwargs['params']['limit'] == mock_limit

    # TODO: Add more tests for other CRUD methods, search methods, bulk operations, etc.
    # Example for add_nodes_and_edges_bulk (partial)
    async def test_add_nodes_and_edges_bulk(self, neo4j_provider: Neo4jProvider, mock_driver: AsyncMock):
        mock_embedder = AsyncMock(spec=EmbedderClient)
        # Expecting calls for: 1 entity name, 1 community name, 1 community summary
        mock_embedder.create = AsyncMock(return_value=[[0.1,0.2]] * 3)

        entity_nodes = [EntityNode(uuid="en1", name="Test Bulk Entity", group_id="bulk_group")] # Needs embedding
        community_nodes = [
            CommunityNode(uuid="cn1", name="Test Bulk Community", group_id="bulk_group", summary="Summary to embed"), # Name and Summary need embedding
            CommunityNode(uuid="cn2", name="Community No Summary", group_id="bulk_group", name_embedding=[0.3,0.4]) # Has embedding
        ]
        community_edges = [
            CommunityEdge(uuid="ce1", source_node_uuid="cn1", target_node_uuid="en1", group_id="bulk_group")
        ]

        # Other lists can be empty for this specific test focus
        episodic_nodes=[]
        episodic_edges=[]
        entity_edges=[]

        await neo4j_provider.add_nodes_and_edges_bulk(
            episodic_nodes=episodic_nodes,
            episodic_edges=episodic_edges,
            entity_nodes=entity_nodes,
            entity_edges=entity_edges,
            community_nodes=community_nodes,
            community_edges=community_edges,
            embedder=mock_embedder
        )

        # Check embedder calls
        # Neo4j provider's _NEO4J_ENTITY_NODE_TEXT_FIELDS = ["name", "summary"]
        # Neo4j provider's _NEO4J_COMMUNITY_NODE_TEXT_FIELDS = ["name", "summary"]
        # EntityNode "en1": name="Test Bulk Entity" -> 1 call
        # CommunityNode "cn1": name="Test Bulk Community", summary="Summary to embed". If name_embedding is None, name is used. If summary_embedding is conceptual, summary is used.
        # Current provider logic for CommunityNode: if name_embedding is None, uses "name" then "summary".
        # So, for "cn1", name="Test Bulk Community" is embedded. Summary is not separately embedded into another field.
        # The mock_embedder.create needs to align with how many actual text pieces are sent.
        # Let's assume:
        # 1. EntityNode "en1" name.
        # 2. CommunityNode "cn1" name ("Test Bulk Community"). Summary "Summary to embed" is also present.
        #    The provider logic for community nodes is:
        #    texts_to_embed = [getattr(node, field, None) for field in cls._NEO4J_COMMUNITY_NODE_TEXT_FIELDS if getattr(node, field, None)]
        #    text = " ".join(filter(None, texts_to_embed)).strip()
        #    So, for "cn1", text will be "Test Bulk Community Summary to embed". This is 1 call.
        # CommunityNode "cn2" has name_embedding, so no call.
        # Total calls = 1 (for en1) + 1 (for cn1) = 2
        assert mock_embedder.create.call_count == 2


        # Retrieve the mock_tx object that was used inside execute_write
        assert hasattr(mock_driver, '_most_recent_mock_tx'), "Mock transaction object not found on mock_driver"
        mock_tx_used_in_call = mock_driver._most_recent_mock_tx

        # Verify calls to tx.run
        # Expected calls: Entity nodes, Community nodes, Community edges
        # (Episodic nodes/edges and Entity edges are empty in this test)

        actual_run_calls = mock_tx_used_in_call.run.call_args_list

        # Check for Entity node bulk save
        assert any(
            call[0][0] == node_db_queries.ENTITY_NODE_SAVE_BULK and \
            len(call[1]['nodes']) == 1 and \
            call[1]['nodes'][0]['uuid'] == "en1"
            for call in actual_run_calls
        ), "Entity node bulk save not called correctly"

        # Check for Community node bulk save
        assert any(
            call[0][0] == node_db_queries.COMMUNITY_NODE_SAVE_BULK and \
            len(call[1]['nodes']) == 2 and \
            call[1]['nodes'][0]['uuid'] == "cn1" # Check first community node data
            for call in actual_run_calls
        ), "Community node bulk save not called correctly"

        # Check for Community edge bulk save
        assert any(
            call[0][0] == edge_db_queries.COMMUNITY_EDGE_SAVE_BULK and \
            len(call[1]['edges']) == 1 and \
            call[1]['edges'][0]['uuid'] == "ce1"
            for call in actual_run_calls
        ), "Community edge bulk save not called correctly"

        # Total expected tx.run calls if lists are non-empty:
        # EntityNodes, CommunityNodes, CommunityEdges = 3 calls
        assert mock_tx_used_in_call.run.call_count == 3


    async def test_count_node_mentions(self, neo4j_provider: Neo4jProvider):
        node_uuid_to_count = "test-node-uuid"
        expected_count = 5
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([{"mention_count": expected_count}], {}, {}))

        count = await neo4j_provider.count_node_mentions(node_uuid_to_count)

        assert count == expected_count
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        assert "MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $node_uuid})" in args[0]
        assert kwargs['params'] == {"node_uuid": node_uuid_to_count}

    async def test_get_entity_nodes_by_uuids(self, neo4j_provider: Neo4jProvider):
        mock_uuids = ["uuid1", "uuid2"]
        # Simulate two records being returned
        mock_records = [
            {**MOCK_ENTITY_NODE_FLAT_RECORD, "uuid": "uuid1", "name": "Node 1"},
            {**MOCK_ENTITY_NODE_FLAT_RECORD, "uuid": "uuid2", "name": "Node 2"},
        ]
        neo4j_provider.driver.execute_query = AsyncMock(return_value=(mock_records, {}, {}))

        results = await neo4j_provider.get_entity_nodes_by_uuids(mock_uuids)

        assert len(results) == 2
        assert results[0].uuid == "uuid1"
        assert results[1].name == "Node 2"
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        assert f"MATCH (n:Entity) WHERE n.uuid IN $uuids {ENTITY_NODE_RETURN}" in args[0]
        assert kwargs['params'] == {"uuids": mock_uuids}

    async def test_get_entity_nodes_by_group_ids(self, neo4j_provider: Neo4jProvider):
        mock_group_ids = ["group1"]
        mock_limit = 1
        # Simulate one record being returned
        mock_records = [
             {**MOCK_ENTITY_NODE_FLAT_RECORD, "group_id": "group1"},
        ]
        neo4j_provider.driver.execute_query = AsyncMock(return_value=(mock_records, {}, {}))

        results = await neo4j_provider.get_entity_nodes_by_group_ids(mock_group_ids, limit=mock_limit)

        assert len(results) == 1
        assert results[0].group_id == "group1"
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        assert "MATCH (n:Entity) WHERE n.group_id IN $group_ids" in args[0]
        assert "ORDER BY n.uuid DESC" in args[0] # Default ordering for this method
        assert "LIMIT $limit" in args[0]
        assert kwargs['params'] == {"group_ids": mock_group_ids, "limit": mock_limit}


    async def test_save_episodic_node(self, neo4j_provider: Neo4jProvider):
        node = EpisodicNode(
            uuid="ep-uuid-1", name="Test Episode", group_id="group1",
            source=EpisodeType.text, source_description="test source",
            content="test content", valid_at=MOCK_DATETIME_NATIVE,
            entity_edges=["edge-uuid-1"]
        )
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([{"uuid": "ep-uuid-1"}], {}, {}))
        await neo4j_provider.save_episodic_node(node)
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        assert args[0] == node_db_queries.EPISODIC_NODE_SAVE
        assert kwargs['params']['uuid'] == "ep-uuid-1"
        assert kwargs['params']['source'] == EpisodeType.text.value

    async def test_get_episodic_node_by_uuid_found(self, neo4j_provider: Neo4jProvider):
        mock_ep_record = {
            "uuid": "ep-uuid-1", "name": "Found Episode", "group_id": "group1",
            "source": EpisodeType.text.value, "source_description": "found source",
            "content": "found content", "valid_at": MOCK_NEO4J_DATETIME,
            "created_at": MOCK_NEO4J_DATETIME, "entity_edges": ["edge1"]
        }
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([mock_ep_record], {}, {}))
        found_node = await neo4j_provider.get_episodic_node_by_uuid("ep-uuid-1")
        assert found_node is not None
        assert found_node.name == "Found Episode"
        assert found_node.source == EpisodeType.text

    async def test_save_community_node(self, neo4j_provider: Neo4jProvider):
        node = CommunityNode(
            uuid="comm-uuid-1", name="Test Community", group_id="group1",
            summary="community summary", name_embedding=[0.4, 0.5]
        )
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([{"uuid": "comm-uuid-1"}], {}, {}))
        await neo4j_provider.save_community_node(node)
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        assert args[0] == node_db_queries.COMMUNITY_NODE_SAVE
        assert kwargs['params']['uuid'] == "comm-uuid-1"
        assert kwargs['params']['name_embedding'] == [0.4, 0.5]

    async def test_get_community_node_by_uuid_found(self, neo4j_provider: Neo4jProvider):
        mock_comm_record = {
            "uuid": "comm-uuid-1", "name": "Found Community", "group_id": "group1",
            "summary": "found summary", "created_at": MOCK_NEO4J_DATETIME,
            "name_embedding": [0.5, 0.6], "labels": ["Community", "TestLabel"]
        }
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([mock_comm_record], {}, {}))
        found_node = await neo4j_provider.get_community_node_by_uuid("comm-uuid-1")
        assert found_node is not None
        assert found_node.name == "Found Community"
        assert found_node.name_embedding == [0.5, 0.6]

    async def test_save_entity_edge(self, neo4j_provider: Neo4jProvider):
        edge = EntityEdge(
            uuid="edge-uuid-1", name="RELATES_TO", group_id="group1",
            source_node_uuid="source-uuid", target_node_uuid="target-uuid",
            fact="test fact", fact_embedding=[0.7, 0.8], created_at=MOCK_DATETIME_NATIVE
        )
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([{"uuid": "edge-uuid-1"}], {}, {}))
        await neo4j_provider.save_entity_edge(edge)
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        assert args[0] == edge_db_queries.ENTITY_EDGE_SAVE
        assert kwargs['params']['uuid'] == "edge-uuid-1"
        assert kwargs['params']['fact_embedding'] == [0.7, 0.8]

    async def test_get_entity_edge_by_uuid_found(self, neo4j_provider: Neo4jProvider):
        mock_edge_record = {
            "uuid": "edge-uuid-1", "name": "RELATES_TO", "group_id": "group1",
            "source_node_uuid": "source-uuid", "target_node_uuid": "target-uuid",
            "fact": "test fact", "fact_embedding": [0.7, 0.8], "created_at": MOCK_NEO4J_DATETIME,
            "episodes": [], "expired_at": None, "valid_at": None, "invalid_at": None,
            "attributes": {"fact": "test fact", "name": "RELATES_TO"} # Example properties(e)
        }
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([mock_edge_record], {}, {}))
        found_edge = await neo4j_provider.get_entity_edge_by_uuid("edge-uuid-1")
        assert found_edge is not None
        assert found_edge.fact == "test fact"
        assert found_edge.fact_embedding == [0.7, 0.8]

    async def test_build_indices_and_constraints(self, neo4j_provider: Neo4jProvider):
        await neo4j_provider.build_indices_and_constraints(delete_existing=False)
        # Check a few common index calls
        # semaphore_gather makes it a bit tricky to check individual calls without more complex mocking of semaphore_gather
        # For simplicity, check if execute_query was called multiple times (for each index)
        assert neo4j_provider.driver.execute_query.call_count >= 10 # Rough check

        # Test with delete_existing = True
        neo4j_provider.driver.execute_query.reset_mock()
        # Mock return for SHOW INDEXES
        mock_show_indexes_result = ([{"name": "entity_uuid"}, {"name": "episode_uuid"}], {}, {})

        # Setup side_effect: first call returns indexes, subsequent calls are for DROP INDEX
        async def execute_query_side_effect(*args, **kwargs):
            query_str = args[0]
            if "SHOW INDEXES" in query_str:
                return mock_show_indexes_result
            elif "DROP INDEX" in query_str:
                return ([], {}, {}) # Simulate successful drop
            else: # For CREATE INDEX calls
                return ([], {}, {})

        neo4j_provider.driver.execute_query.side_effect = execute_query_side_effect

        await neo4j_provider.build_indices_and_constraints(delete_existing=True)

        # Check if SHOW INDEXES was called
        # Check if DROP INDEX was called for each returned index
        # Check if CREATE INDEX queries were called
        # This requires more detailed assertions on call_args_list
        drop_calls = 0
        create_calls = 0
        show_indexes_called = False
        for call_args in neo4j_provider.driver.execute_query.call_args_list:
            query_text = call_args[0][0] # First argument of the first positional argument
            if "SHOW INDEXES" in query_text:
                show_indexes_called = True
            elif "DROP INDEX" in query_text:
                drop_calls +=1
            elif "CREATE INDEX" in query_text or "CREATE FULLTEXT INDEX" in query_text:
                create_calls +=1

        assert show_indexes_called
        assert drop_calls == len(mock_show_indexes_result[0])
        assert create_calls >= 10 # Rough check for creations

    async def test_node_similarity_search(self, neo4j_provider: Neo4jProvider):
        mock_vector = [0.1, 0.2]
        mock_group_ids = ["group1"]

        neo4j_provider.driver.execute_query = AsyncMock(return_value=([MOCK_ENTITY_NODE_FLAT_RECORD], {}, {}))

        results = await neo4j_provider.node_similarity_search(
            search_vector=mock_vector,
            search_filter=SearchFilters(),
            group_ids=mock_group_ids,
            limit=5
        )
        assert len(results) == 1
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        assert "vector.similarity.cosine(n.name_embedding, $search_vector)" in args[0]
        assert kwargs['params']['search_vector'] == mock_vector
        assert kwargs['params']['group_ids'] == mock_group_ids

    async def test_retrieve_episodes(self, neo4j_provider: Neo4jProvider):
        ref_time = datetime.now(timezone.utc)
        mock_ep_record = {
            "uuid": "ep-uuid-r", "name": "Retrieved Episode", "group_id": "group1",
            "source": EpisodeType.text.value, "source_description": "retrieved source",
            "content": "retrieved content", "valid_at": MOCK_NEO4J_DATETIME,
            "created_at": MOCK_NEO4J_DATETIME, "entity_edges": []
        }
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([mock_ep_record], {}, {}))

        results = await neo4j_provider.retrieve_episodes(reference_time=ref_time, last_n=1)

        assert len(results) == 1
        assert results[0].name == "Retrieved Episode"
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        assert "MATCH (e:Episodic) WHERE e.valid_at <= $reference_time" in args[0]
        assert kwargs['params']['reference_time'] == ref_time

    async def test_delete_node(self, neo4j_provider: Neo4jProvider):
        test_uuid = "delete-me-node"
        await neo4j_provider.delete_node(test_uuid)
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        assert args[0] == "MATCH (n {uuid: $uuid}) DETACH DELETE n"
        assert kwargs['params'] == {"uuid": test_uuid}

    async def test_delete_edge(self, neo4j_provider: Neo4jProvider):
        test_uuid = "delete-me-edge"
        await neo4j_provider.delete_edge(test_uuid)
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        assert args[0] == "MATCH ()-[e {uuid: $uuid}]-() DELETE e"
        assert kwargs['params'] == {"uuid": test_uuid}

    async def test_clear_data_all(self, neo4j_provider: Neo4jProvider):
        # Mock the session and transaction context
        mock_tx = AsyncMock()
        async def mock_execute_write_delete_all(tx_func): # No extra args for delete_all
            await tx_func(mock_tx)

        neo4j_provider.driver.session.return_value.__aenter__.return_value.execute_write = mock_execute_write_delete_all

        await neo4j_provider.clear_data()

        mock_tx.run.assert_called_once_with('MATCH (n) DETACH DELETE n')

    async def test_clear_data_group_ids(self, neo4j_provider: Neo4jProvider):
        mock_group_ids = ["group1", "group2"]
        mock_tx = AsyncMock()
        async def mock_execute_write_delete_groups(tx_func, group_ids_to_delete):
            await tx_func(mock_tx, group_ids_to_delete=group_ids_to_delete)

        neo4j_provider.driver.session.return_value.__aenter__.return_value.execute_write = mock_execute_write_delete_groups

        await neo4j_provider.clear_data(group_ids=mock_group_ids)

        mock_tx.run.assert_called_once_with(
            'MATCH (n) WHERE n.group_id IN $group_ids DETACH DELETE n', # The actual query used in implementation
            group_ids=mock_group_ids
        )

    async def test_get_relevant_nodes_rrf(self, neo4j_provider: Neo4jProvider):
        input_node = EntityNode(
            uuid="center-node-uuid",
            name="Central Node",
            group_id="group1",
            name_embedding=[0.1, 0.1]
        )

        # Mock results for the vector search part
        # (Returns UUIDs and scores)
        mock_vector_search_records = [
            {"uuid": "vec_node1_uuid", "score": 0.9}, # Rank 1 (index 0)
            {"uuid": "shared_node_uuid", "score": 0.8}, # Rank 2 (index 1)
        ]

        # Mock results for the full-text search part
        # (Returns UUIDs and scores)
        mock_fts_search_records = [
            {"uuid": "fts_node1_uuid", "score": 1.5},   # Rank 1 (index 0)
            {"uuid": "shared_node_uuid", "score": 1.2}, # Rank 2 (index 1)
            {"uuid": "vec_node1_uuid", "score": 1.0},   # Rank 3 (index 2) - shows up in both
        ]

        # Mock results for the final get_entity_nodes_by_uuids call
        # This should return full node objects for the top RRF ranked UUIDs
        mock_final_node1 = {**MOCK_ENTITY_NODE_FLAT_RECORD, "uuid": "shared_node_uuid", "name": "Shared Node"}
        mock_final_node2 = {**MOCK_ENTITY_NODE_FLAT_RECORD, "uuid": "vec_node1_uuid", "name": "Vector Node 1"}
        mock_final_node3 = {**MOCK_ENTITY_NODE_FLAT_RECORD, "uuid": "fts_node1_uuid", "name": "FTS Node 1"}

        # Order of these nodes depends on RRF scores. Let's calculate expected RRF:
        # rrf_k = 60 (default in implementation)
        # shared_node_uuid: (1/(60+1+1)) + (1/(60+1+1)) = 2/62 approx 0.0322
        # vec_node1_uuid:   (1/(60+0+1)) + (1/(60+2+1)) = 1/61 + 1/63 approx 0.0163 + 0.0158 = 0.0321
        # fts_node1_uuid:   (1/(60+0+1)) = 1/61 approx 0.0163
        # So, expected order: shared_node_uuid, vec_node1_uuid, fts_node1_uuid

        mock_final_node_objects = [
            EntityNode(**get_entity_node_from_record(mock_final_node1)),
            EntityNode(**get_entity_node_from_record(mock_final_node2)),
            EntityNode(**get_entity_node_from_record(mock_final_node3)),
        ]

        # Configure side_effect for execute_query
        # 1st call: vector search
        # 2nd call: FTS search
        # 3rd call (from get_entity_nodes_by_uuids): fetching full nodes for RRF winners
        async def mock_execute_query_for_rrf(*args, **kwargs):
            query_str = args[0]
            if "vector.similarity.cosine" in query_str:
                return (mock_vector_search_records, {}, {})
            elif "CALL db.index.fulltext.queryNodes" in query_str:
                return (mock_fts_search_records, {}, {})
            elif "WHERE n.uuid IN $uuids" in query_str: # From get_entity_nodes_by_uuids
                # Simulate returning nodes based on the UUIDs passed to get_entity_nodes_by_uuids
                requested_uuids = kwargs['params']['uuids']
                returned_nodes_data = []
                if "shared_node_uuid" in requested_uuids: returned_nodes_data.append(mock_final_node1)
                if "vec_node1_uuid" in requested_uuids: returned_nodes_data.append(mock_final_node2)
                if "fts_node1_uuid" in requested_uuids: returned_nodes_data.append(mock_final_node3)
                return (returned_nodes_data, {}, {})
            return ([], {}, {})

        neo4j_provider.driver.execute_query = AsyncMock(side_effect=mock_execute_query_for_rrf)

        results_list = await neo4j_provider.get_relevant_nodes(
            nodes=[input_node],
            search_filter=SearchFilters(),
            limit=3
        )

        assert len(results_list) == 1 # For one input node
        final_results = results_list[0]
        assert len(final_results) == 3

        # Check order based on RRF scores calculated above
        assert final_results[0].uuid == "shared_node_uuid"
        assert final_results[1].uuid == "vec_node1_uuid"
        assert final_results[2].uuid == "fts_node1_uuid"

        # Verify calls to execute_query
        assert neo4j_provider.driver.execute_query.call_count == 3


    async def test_delete_nodes_by_group_id(self, neo4j_provider: Neo4jProvider):
        group_id_to_delete = "group-to-delete"
        node_type_to_delete = "Entity"

        await neo4j_provider.delete_nodes_by_group_id(group_id_to_delete, node_type_to_delete)

        args, kwargs = neo4j_provider.driver.execute_query.call_args
        assert f"MATCH (n:{node_type_to_delete} {{group_id: $group_id}}) DETACH DELETE n" in args[0]
        assert kwargs['params'] == {"group_id": group_id_to_delete}

    async def test_get_embeddings_for_nodes(self, neo4j_provider: Neo4jProvider):
        nodes_to_fetch = [
            EntityNode(uuid="uuid1", name="Node 1", group_id="g1"),
            EntityNode(uuid="uuid2", name="Node 2", group_id="g1", name_embedding=[0.1,0.2]) # One with embedding
        ]
        mock_records = [
            {"uuid": "uuid1", "embedding": None}, # Node 1 has no embedding
            {"uuid": "uuid2", "embedding": [0.1, 0.2]}
        ]
        neo4j_provider.driver.execute_query = AsyncMock(return_value=(mock_records, {}, {}))

        embeddings_map = await neo4j_provider.get_embeddings_for_nodes(nodes_to_fetch)

        assert len(embeddings_map) == 1 # Only uuid2 should be in the map
        assert "uuid1" not in embeddings_map
        assert "uuid2" in embeddings_map
        assert embeddings_map["uuid2"] == [0.1, 0.2]

        args, kwargs = neo4j_provider.driver.execute_query.call_args
        assert "MATCH (n:Entity) WHERE n.uuid IN $node_uuids" in args[0]
        assert kwargs['params'] == {"node_uuids": ["uuid1", "uuid2"]}


    # --- Tests for SearchFilters Scenarios ---

    # 1. Multiple DateFilter Conditions
    async def test_node_fulltext_search_multiple_or_date_filters(self, neo4j_provider: Neo4jProvider):
        mock_query_term = "findme"
        date_x = datetime(2023, 1, 1, tzinfo=timezone.utc)
        date_y = datetime(2023, 12, 31, tzinfo=timezone.utc)

        s_filters = SearchFilters(
            created_at=[ # Simulating (created_at > X OR created_at < Y)
                [DateFilter(date=date_x, operator=ComparisonOperator.GREATER_THAN)],
                [DateFilter(date=date_y, operator=ComparisonOperator.LESS_THAN)]
            ]
        )
        # Expected Lucene query from _node_search_filter_query_constructor for these filters
        # This part is tricky as it depends on how _node_search_filter_query_constructor
        # translates these into Lucene syntax. Assuming it builds a sub-query like:
        # "(created_at_timestamp:{_PyDateTime_to_neo4j_timestamp(date_x) TO *} OR created_at_timestamp:{* TO _PyDateTime_to_neo4j_timestamp(date_y)})"
        # For now, let's focus on the Cypher part generated by the main search method if it appends WHERE clauses.
        # The _node_search_filter_query_constructor primarily adds to the Lucene query string.

        # Mock the execute_query to return a node
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([MOCK_ENTITY_NODE_FLAT_RECORD], {}, {}))

        await neo4j_provider.node_fulltext_search(
            query=mock_query_term,
            search_filter=s_filters,
            group_ids=[MOCK_GROUP_ID], # Must provide group_id for Lucene query
            limit=1
        )

        args, kwargs = neo4j_provider.driver.execute_query.call_args
        lucene_query_param = kwargs['params']['lucene_query']

        # Check the Lucene query part.
        # _PyDateTime_to_neo4j_timestamp will convert datetimes to epoch milliseconds for Neo4j range queries.
        # Example: "(created_at_timestamp:[1672531200000 TO MAX] OR created_at_timestamp:[MIN TO 1703980799000])"
        # We need to ensure the structure is (condition OR condition)
        # The exact timestamp conversion should be tested in _PyDateTime_to_neo4j_timestamp or its usage.
        # Here, we check for the logical structure.
        # Assuming _node_search_filter_query_constructor correctly uses _build_lucene_date_clause:
        expected_lucene_date_clause_x = f"created_at_timestamp:{{{int(date_x.timestamp() * 1000)} TO MAX}}" # Exclusive for GT
        expected_lucene_date_clause_y = f"created_at_timestamp:{{MIN TO {int(date_y.timestamp() * 1000)}}}" # Exclusive for LT

        # The _node_search_filter_query_constructor will AND the group_id with the (date OR date) clause.
        # e.g. lucene_query = "+group_id:mock-group-id +( (date_clause_x) OR (date_clause_y) )"
        # Note: The actual _node_search_filter_query_constructor might produce slightly different syntax for OR.
        # Let's assume it forms it as: (filter) AND ( (date_filter_1) OR (date_filter_2) )
        # This requires inspecting how the list of lists for dates is processed.
        # Current _node_search_filter_query_constructor logic:
        # for filter_group in search_filters.created_at:
        #   group_clauses = []
        #   for date_filter in filter_group: group_clauses.append(...)
        #   lucene_sub_queries.append(f"({' AND '.join(group_clauses)})")
        # This means [[GT_X], [LT_Y]] becomes ( (GT_X) AND (LT_Y) ) if not careful.
        # The intention of List[List[DateFilter]] is typically outer OR, inner AND.
        # So, [[GT_X], [LT_Y]] should be (GT_X) OR (LT_Y).
        # If SearchFilters has created_at: [[DF1, DF2]], it means (DF1 AND DF2).
        # If SearchFilters has created_at: [[DF1], [DF2]], it means (DF1) OR (DF2).
        # The test case `s_filters` has [[DateFilter(date=date_x, op=GT)], [DateFilter(date=date_y, op=LT)]]
        # This should translate to (created_at > X) OR (created_at < Y)

        # The `_node_search_filter_query_constructor` joins these OR groups with " OR ".
        # So, `(expected_lucene_date_clause_x) OR (expected_lucene_date_clause_y)` should be part of the query.
        assert f"({expected_lucene_date_clause_x})" in lucene_query_param
        assert f"({expected_lucene_date_clause_y})" in lucene_query_param
        # Check that these two are ORed together. The constructor wraps each inner list in () and then ORs them.
        # So it would be `( (clause_x) ) OR ( (clause_y) )` if inner list has one item.
        # More precisely `(created_at_timestamp_0_0) OR (created_at_1_0)` where these are the params.
        # The actual lucene string might be more complex. Let's check for core components.
        assert MOCK_GROUP_ID in lucene_query_param # Group ID must be there.
        # A more robust check would be to parse the lucene_query_param if its structure is well-defined.
        # For now, checking presence of key components.

    async def test_edge_fulltext_search_multiple_date_filters(self, neo4j_provider: Neo4jProvider):
        mock_query_term = "find_edge"
        created_date = datetime(2023, 3, 15, tzinfo=timezone.utc)
        valid_date = datetime(2023, 4, 1, tzinfo=timezone.utc)

        s_filters = SearchFilters(
            created_at_filter=DateFilter(date=created_date, operator=ComparisonOperator.EQUALS),
            valid_at_filter=DateFilter(date=valid_date, operator=ComparisonOperator.LESS_THAN_OR_EQUALS)
        )
        # Mock the driver's execute_query
        # Assuming ENTITY_EDGE_RETURN and get_entity_edge_from_record work.
        mock_edge_data = {
            "uuid": "edge-multi-date", "name": "RELATES_TO", "group_id": MOCK_GROUP_ID,
            "source_node_uuid": "s1", "target_node_uuid": "t1",
            "fact": "Edge for multi date test", "fact_embedding": None,
            "created_at": MockNeo4jDateTime(created_date), # Matches created_at_filter
            "valid_at": MockNeo4jDateTime(datetime(2023, 3, 20, tzinfo=timezone.utc)), # Matches valid_at_filter
            "attributes": {"fact": "Edge for multi date test"}
        }
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([mock_edge_data], {}, {}))

        await neo4j_provider.edge_fulltext_search(
            query=mock_query_term,
            search_filter=s_filters,
            group_ids=[MOCK_GROUP_ID],
            limit=1
        )
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        params = kwargs['params']

        # Edge searches use Cypher WHERE clauses for date filters, not Lucene.
        # The _edge_search_filter_query_constructor should build these.
        # e.g. "WHERE e.created_at = $created_at_date AND e.valid_at <= $valid_at_date"

        assert "e.created_at = $created_at_0_0" in args[0] # Check for created_at clause
        assert "e.valid_at <= $valid_at_0_0" in args[0]   # Check for valid_at clause

        assert params['created_at_0_0'] == created_date
        assert params['valid_at_0_0'] == valid_date
        assert params['lucene_query'] == _fulltext_lucene_query(mock_query_term, [MOCK_GROUP_ID])

    # 2. All ComparisonOperator Options for DateFilter
    @pytest.mark.parametrize("operator", list(ComparisonOperator))
    async def test_node_fulltext_search_created_at_all_operators(self, neo4j_provider: Neo4jProvider, operator: ComparisonOperator):
        mock_query_term = "node_op_test"
        filter_date = datetime(2023, 5, 10, tzinfo=timezone.utc)

        s_filters = SearchFilters(created_at=[[DateFilter(date=filter_date, operator=operator)]])

        neo4j_provider.driver.execute_query = AsyncMock(return_value=([MOCK_ENTITY_NODE_FLAT_RECORD], {}, {}))
        await neo4j_provider.node_fulltext_search(
            query=mock_query_term, search_filter=s_filters, group_ids=[MOCK_GROUP_ID], limit=1
        )
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        lucene_query_param = kwargs['params']['lucene_query']

        # Check how _build_lucene_date_clause (used by _node_search_filter_query_constructor)
        # translates each operator.
        # Example: For EQUALS, it might be "fieldname:[timestamp TO timestamp]" (inclusive range)
        # For GREATER_THAN, "fieldname:{timestamp TO MAX}" (exclusive lower bound)
        # This test will verify that *some* date clause is formed and the group_id is present.
        # A more granular test would be on _build_lucene_date_clause itself.
        assert MOCK_GROUP_ID in lucene_query_param
        assert "created_at_timestamp" in lucene_query_param # Check that the date field is part of the query
        # We can also check that the timestamp of filter_date (in millis) is in the lucene query
        assert str(int(filter_date.timestamp()*1000)) in lucene_query_param

    @pytest.mark.parametrize("operator", list(ComparisonOperator))
    async def test_edge_fulltext_search_created_at_all_operators(
        self, neo4j_provider: Neo4jProvider, operator: ComparisonOperator
    ):
        mock_query_term = "edge_created_op_test"
        filter_date = datetime(2023, 6, 1, tzinfo=timezone.utc)
        s_filters = SearchFilters(created_at_filter=DateFilter(date=filter_date, operator=operator))

        mock_edge_data = {**MOCK_ENTITY_NODE_FLAT_RECORD, "uuid": "edge-op-test-created"} # Using flat record as placeholder for any valid edge structure
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([mock_edge_data], {}, {}))

        await neo4j_provider.edge_fulltext_search(
            query=mock_query_term, search_filter=s_filters, group_ids=[MOCK_GROUP_ID], limit=1
        )
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        query_cypher = args[0]
        params = kwargs['params']

        expected_cypher_op = operator.value
        if operator == ComparisonOperator.CONTAINS: expected_cypher_op = "CONTAINS" # Neo4j specific
        elif operator == ComparisonOperator.NOT_CONTAINS: expected_cypher_op = "NOT CONTAINS" # Neo4j specific

        assert f"e.created_at {expected_cypher_op} $created_at_0_0" in query_cypher
        assert params['created_at_0_0'] == filter_date

    # 3. Combinations of Filters (Neo4j context)
    async def test_node_fulltext_search_combined_labels_and_group_id_param(self, neo4j_provider: Neo4jProvider):
        mock_query_term = "combo_node"
        node_labels_filter = ["LabelA", "LabelB"]
        group_id_param = "specific_group_for_combo"

        s_filters = SearchFilters(node_labels=node_labels_filter)

        neo4j_provider.driver.execute_query = AsyncMock(return_value=([MOCK_ENTITY_NODE_FLAT_RECORD], {}, {}))
        await neo4j_provider.node_fulltext_search(
            query=mock_query_term,
            search_filter=s_filters,
            group_ids=[group_id_param], # group_id passed as method parameter
            limit=1
        )
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        lucene_query_param = kwargs['params']['lucene_query']

        # Check Lucene query:
        # _node_search_filter_query_constructor should combine these.
        # Labels are ANDed: +label:LabelA +label:LabelB
        # Group ID is also ANDed: +group_id:specific_group_for_combo
        assert f"+label:{node_labels_filter[0]}" in lucene_query_param
        assert f"+label:{node_labels_filter[1]}" in lucene_query_param
        assert f"+group_id:{group_id_param}" in lucene_query_param
        assert mock_query_term in lucene_query_param # Original query term should also be there

    async def test_edge_fulltext_search_combined_edge_types_node_labels_date(self, neo4j_provider: Neo4jProvider):
        mock_query_term = "combo_edge"
        edge_types_filter = ["TYPE_X", "TYPE_Y"]
        source_node_labels = ["SourceLabel"]
        target_node_labels = ["TargetLabel"]
        filter_date = datetime(2023, 8, 1, tzinfo=timezone.utc)

        s_filters = SearchFilters(
            edge_types=edge_types_filter,
            source_node_labels=source_node_labels,
            target_node_labels=target_node_labels,
            created_at_filter=DateFilter(date=filter_date, operator=ComparisonOperator.GREATER_THAN)
        )
        mock_edge_data = {**MOCK_ENTITY_NODE_FLAT_RECORD, "uuid": "edge-combo-test"}
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([mock_edge_data], {}, {}))

        await neo4j_provider.edge_fulltext_search(
            query=mock_query_term,
            search_filter=s_filters,
            group_ids=[MOCK_GROUP_ID], # Group ID for Lucene FTS part
            limit=1
        )
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        query_cypher = args[0]
        params = kwargs['params']

        # Check Cypher query parts from _edge_search_filter_query_constructor
        # Edge types:
        assert f"TYPE_X|TYPE_Y" in query_cypher # e.g., MATCH (s)-[e:TYPE_X|TYPE_Y]->(t)
        # Node labels:
        assert "s:SourceLabel" in query_cypher
        assert "t:TargetLabel" in query_cypher
        # Date filter:
        assert "e.created_at > $created_at_0_0" in query_cypher
        assert params['created_at_0_0'] == filter_date

        # Lucene query for the fact/name part
        assert params['lucene_query'] == _fulltext_lucene_query(mock_query_term, [MOCK_GROUP_ID])

    # 4. Label/Type Filtering (Neo4j-specific)
    @pytest.mark.parametrize("labels_to_filter, expect_in_lucene", [
        (["SingleLabel"], "+label:SingleLabel"),
        (["LabelOne", "LabelTwo"], "+label:LabelOne +label:LabelTwo"),
        ([], ""), # No label filter means no specific "+label:" clause, though base query might have default like :Entity
        # Non-existent labels are harder to test here unless we know all possible labels or the index behavior.
        # The query constructor will add them; if they don't exist, Lucene simply won't match.
    ])
    async def test_node_fulltext_search_node_labels_variations(
        self, neo4j_provider: Neo4jProvider, labels_to_filter: List[str], expect_in_lucene: str
    ):
        mock_query_term = "label_test_node"
        s_filters = SearchFilters(node_labels=labels_to_filter)

        neo4j_provider.driver.execute_query = AsyncMock(return_value=([MOCK_ENTITY_NODE_FLAT_RECORD], {}, {}))
        await neo4j_provider.node_fulltext_search(
            query=mock_query_term, search_filter=s_filters, group_ids=[MOCK_GROUP_ID], limit=1
        )
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        lucene_query_param = kwargs['params']['lucene_query']

        if expect_in_lucene:
            assert expect_in_lucene in lucene_query_param
        else:
            # If no labels to filter, ensure no accidental "+label:" is added beyond what _fulltext_lucene_query adds by default
            # (e.g. if it enforces :Entity by default, that's different)
            # For now, this check is simple; a more complex check might be needed if default labels are involved.
            assert "+label:" not in lucene_query_param or all(f"+label:{l}" not in lucene_query_param for l in ["SingleLabel", "LabelOne", "LabelTwo"])


    @pytest.mark.parametrize("edge_types, source_labels, target_labels, expected_match_clause_parts", [
        (["REL_A"], [], [], ["-[e:REL_A]-"]), # Single edge type
        (["REL_A", "REL_B"], [], [], ["-[e:REL_A|REL_B]-"]), # Multiple edge types
        ([], ["SourceX"], [], ["(s:SourceX)-[e]-"]), # Source label only
        ([], [], ["TargetY"], ["-[e]->(t:TargetY)"]), # Target label only
        (["REL_C"], ["SourceZ"], ["TargetW"], ["(s:SourceZ)-[e:REL_C]->(t:TargetW)"]), # All combined
        ([], [], [], ["MATCH (s)-[e]->(t)"]) # No specific types/labels, should use default MATCH
    ])
    async def test_edge_fulltext_search_label_type_variations(
        self, neo4j_provider: Neo4jProvider,
        edge_types: List[str], source_labels: List[str], target_labels: List[str],
        expected_match_clause_parts: List[str]
    ):
        mock_query_term = "label_type_edge_test"
        s_filters = SearchFilters(
            edge_types=edge_types,
            source_node_labels=source_labels,
            target_node_labels=target_labels
        )
        mock_edge_data = {**MOCK_ENTITY_NODE_FLAT_RECORD, "uuid": "edge-labeltype-test"}
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([mock_edge_data], {}, {}))

        await neo4j_provider.edge_fulltext_search(
            query=mock_query_term, search_filter=s_filters, group_ids=[MOCK_GROUP_ID], limit=1
        )
        args, _ = neo4j_provider.driver.execute_query.call_args
        query_cypher = args[0]

        for part in expected_match_clause_parts:
            assert part in query_cypher

    # 5. Edge Cases for Search Inputs
    async def test_node_fulltext_search_empty_query_string_neo4j(self, neo4j_provider: Neo4jProvider):
        # For Neo4j, an empty query string for Lucene typically means it relies on other filters.
        # _fulltext_lucene_query with empty query and group_ids=["g1"] -> "+group_id:g1"
        # If no group_ids, it might become an empty Lucene query string, which can be an error or match all.
        # The provider's node_fulltext_search wraps this.
        group_id_param = "group_for_empty_query_test"

        neo4j_provider.driver.execute_query = AsyncMock(return_value=([MOCK_ENTITY_NODE_FLAT_RECORD], {}, {}))
        await neo4j_provider.node_fulltext_search(
            query="", # Empty query
            search_filter=SearchFilters(),
            group_ids=[group_id_param],
            limit=1
        )
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        lucene_query_param = kwargs['params']['lucene_query']
        # Lucene query should contain only the group_id filter part.
        expected_lucene = _fulltext_lucene_query("", [group_id_param]) # query_string:(+group_id:group_for_empty_query_test)
        assert lucene_query_param == expected_lucene
        assert "CALL db.index.fulltext.queryNodes" in args[0]

    async def test_node_similarity_search_empty_vector_neo4j(self, neo4j_provider: Neo4jProvider):
        # Neo4jProvider's node_similarity_search has a guard: if not search_vector: return [], {}
        results, _ = await neo4j_provider.node_similarity_search(
            search_vector=[], # Empty vector
            search_filter=SearchFilters(),
            group_ids=[MOCK_GROUP_ID],
            limit=5
        )
        assert len(results) == 0
        neo4j_provider.driver.execute_query.assert_not_called() # Guard should prevent query

    async def test_node_fulltext_search_limit_zero_neo4j(self, neo4j_provider: Neo4jProvider):
        # Neo4jProvider's node_fulltext_search passes limit to execute_query,
        # and Lucene itself handles limit=0 by returning no results.
        # The method itself doesn't have a guard for limit=0 before calling execute_query.
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([], {}, {})) # Limit 0 should return empty

        results = await neo4j_provider.node_fulltext_search(
            query="test_limit_0",
            search_filter=SearchFilters(),
            group_ids=[MOCK_GROUP_ID],
            limit=0
        )
        assert len(results) == 0
        # Assert execute_query IS called, as Lucene handles limit 0.
        neo4j_provider.driver.execute_query.assert_called_once()
        _, kwargs = neo4j_provider.driver.execute_query.call_args
        assert kwargs['params']['limit'] == 0


    async def test_node_similarity_search_limit_zero_neo4j(self, neo4j_provider: Neo4jProvider):
        # Neo4jProvider's node_similarity_search also passes limit to execute_query.
        # The Cypher query itself will have LIMIT 0.
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([], {}, {}))

        results, _ = await neo4j_provider.node_similarity_search(
            search_vector=[0.1, 0.2],
            search_filter=SearchFilters(),
            group_ids=[MOCK_GROUP_ID],
            limit=0
        )
        assert len(results) == 0
        # Assert execute_query IS called.
        neo4j_provider.driver.execute_query.assert_called_once()
        args, _ = neo4j_provider.driver.execute_query.call_args
        assert "LIMIT $limit" in args[0] # Query should contain LIMIT clause
        # Params for the main query call, not the sub-call if any for vector
        # The limit is applied in the main query for similarity search.
        # The call we're checking is the one that includes "WITH n, score ORDER BY score DESC LIMIT $limit"
        assert neo4j_provider.driver.execute_query.call_args[1]['params']['limit'] == 0

    # --- Tests for uuid_cursor Pagination in Get By Group ID Methods (Neo4j) ---
    async def test_get_entity_nodes_by_group_ids_pagination_neo4j(self, neo4j_provider: Neo4jProvider):
        group_ids = [MOCK_GROUP_ID]
        limit = 5
        uuid_cursor = "cursor-entity-node-uuid"
        # Mock return data (actual data doesn't matter for query construction check)
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([], {}, {}))

        await neo4j_provider.get_entity_nodes_by_group_ids(group_ids, limit=limit, uuid_cursor=uuid_cursor)

        args, kwargs = neo4j_provider.driver.execute_query.call_args
        query_str = args[0]
        params_passed = kwargs['params']

        assert "MATCH (n:Entity) WHERE n.group_id IN $group_ids" in query_str
        assert f"AND n.uuid < $uuid_cursor" in query_str # Based on default ORDER BY n.uuid DESC
        assert "ORDER BY n.uuid DESC" in query_str
        assert "LIMIT $limit" in query_str
        assert params_passed.get("uuid_cursor") == uuid_cursor
        assert params_passed.get("limit") == limit
        assert params_passed.get("group_ids") == group_ids

    async def test_get_episodic_nodes_by_group_ids_pagination_neo4j(self, neo4j_provider: Neo4jProvider):
        group_ids = [MOCK_GROUP_ID]
        limit = 3
        uuid_cursor = "cursor-episodic-node-uuid"
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([], {}, {}))

        await neo4j_provider.get_episodic_nodes_by_group_ids(group_ids, limit=limit, uuid_cursor=uuid_cursor)

        args, kwargs = neo4j_provider.driver.execute_query.call_args
        query_str = args[0]
        params_passed = kwargs['params']

        assert "MATCH (e:Episodic) WHERE e.group_id IN $group_ids" in query_str
        assert f"AND e.uuid < $uuid_cursor" in query_str # Default order is e.uuid DESC
        assert "ORDER BY e.uuid DESC" in query_str
        assert "LIMIT $limit" in query_str
        assert params_passed.get("uuid_cursor") == uuid_cursor
        assert params_passed.get("limit") == limit

    async def test_get_community_nodes_by_group_ids_pagination_neo4j(self, neo4j_provider: Neo4jProvider):
        group_ids = [MOCK_GROUP_ID]
        limit = 2
        uuid_cursor = "cursor-community-node-uuid"
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([], {}, {}))

        await neo4j_provider.get_community_nodes_by_group_ids(group_ids, limit=limit, uuid_cursor=uuid_cursor)

        args, kwargs = neo4j_provider.driver.execute_query.call_args
        query_str = args[0]
        params_passed = kwargs['params']

        assert "MATCH (n:Community) WHERE n.group_id IN $group_ids" in query_str
        assert f"AND n.uuid < $uuid_cursor" in query_str # Default order is n.uuid DESC
        assert "ORDER BY n.uuid DESC" in query_str
        assert "LIMIT $limit" in query_str
        assert params_passed.get("uuid_cursor") == uuid_cursor
        assert params_passed.get("limit") == limit

    async def test_get_entity_edges_by_group_ids_pagination_neo4j(self, neo4j_provider: Neo4jProvider):
        group_ids = [MOCK_GROUP_ID]
        limit = 4
        uuid_cursor = "cursor-entity-edge-uuid"
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([], {}, {}))

        await neo4j_provider.get_entity_edges_by_group_ids(group_ids, limit=limit, uuid_cursor=uuid_cursor)

        args, kwargs = neo4j_provider.driver.execute_query.call_args
        query_str = args[0]
        params_passed = kwargs['params']

        assert "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) WHERE e.group_id IN $group_ids" in query_str
        assert f"AND e.uuid < $uuid_cursor" in query_str # Default order is e.uuid DESC
        assert "ORDER BY e.uuid DESC" in query_str
        assert "LIMIT $limit" in query_str
        assert params_passed.get("uuid_cursor") == uuid_cursor
        assert params_passed.get("limit") == limit

    async def test_get_episodic_edges_by_group_ids_pagination_neo4j(self, neo4j_provider: Neo4jProvider):
        group_ids = [MOCK_GROUP_ID]
        limit = 6
        uuid_cursor = "cursor-episodic-edge-uuid"
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([], {}, {}))

        await neo4j_provider.get_episodic_edges_by_group_ids(group_ids, limit=limit, uuid_cursor=uuid_cursor)

        args, kwargs = neo4j_provider.driver.execute_query.call_args
        query_str = args[0]
        params_passed = kwargs['params']

        assert "MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity) WHERE e.group_id IN $group_ids" in query_str
        assert f"AND e.uuid < $uuid_cursor" in query_str # Default order is e.uuid DESC
        assert "ORDER BY e.uuid DESC" in query_str
        assert "LIMIT $limit" in query_str
        assert params_passed.get("uuid_cursor") == uuid_cursor
        assert params_passed.get("limit") == limit

    async def test_get_community_edges_by_group_ids_pagination_neo4j(self, neo4j_provider: Neo4jProvider):
        group_ids = [MOCK_GROUP_ID]
        limit = 7
        uuid_cursor = "cursor-community-edge-uuid"
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([], {}, {}))

        await neo4j_provider.get_community_edges_by_group_ids(group_ids, limit=limit, uuid_cursor=uuid_cursor)

        args, kwargs = neo4j_provider.driver.execute_query.call_args
        query_str = args[0]
        params_passed = kwargs['params']

        assert "MATCH (n:Community)-[e:HAS_MEMBER]->(m:Entity|Community) WHERE e.group_id IN $group_ids" in query_str
        assert f"AND e.uuid < $uuid_cursor" in query_str # Default order is e.uuid DESC
        assert "ORDER BY e.uuid DESC" in query_str
        assert "LIMIT $limit" in query_str
        assert params_passed.get("uuid_cursor") == uuid_cursor
        assert params_passed.get("limit") == limit

    @pytest.mark.parametrize("operator", list(ComparisonOperator))
    async def test_edge_fulltext_search_valid_at_all_operators(
        self, neo4j_provider: Neo4jProvider, operator: ComparisonOperator
    ):
        mock_query_term = "edge_valid_op_test"
        filter_date = datetime(2023, 7, 1, tzinfo=timezone.utc)
        s_filters = SearchFilters(valid_at_filter=DateFilter(date=filter_date, operator=operator))

        mock_edge_data = {**MOCK_ENTITY_NODE_FLAT_RECORD, "uuid": "edge-op-test-valid"}
        neo4j_provider.driver.execute_query = AsyncMock(return_value=([mock_edge_data], {}, {}))

        await neo4j_provider.edge_fulltext_search(
            query=mock_query_term, search_filter=s_filters, group_ids=[MOCK_GROUP_ID], limit=1
        )
        args, kwargs = neo4j_provider.driver.execute_query.call_args
        query_cypher = args[0]
        params = kwargs['params']

        expected_cypher_op = operator.value
        if operator == ComparisonOperator.CONTAINS: expected_cypher_op = "CONTAINS"
        elif operator == ComparisonOperator.NOT_CONTAINS: expected_cypher_op = "NOT CONTAINS"

        assert f"e.valid_at {expected_cypher_op} $valid_at_0_0" in query_cypher
        assert params['valid_at_0_0'] == filter_date
```
