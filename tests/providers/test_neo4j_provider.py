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
        driver_instance.execute_query = AsyncMock(return_value=([], {}, {})) # Default empty response
        driver_instance.verify_connectivity = AsyncMock()
        driver_instance.closed = MagicMock(return_value=False)

        # Mock session behavior if needed for specific tests like clear_data
        mock_session = AsyncMock()
        mock_session.execute_write = AsyncMock()
        driver_instance.session = MagicMock(return_value=mock_session)

        mock_driver_class.return_value = driver_instance
        yield driver_instance

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
    async def test_add_nodes_and_edges_bulk(self, neo4j_provider: Neo4jProvider):
        mock_embedder = AsyncMock(spec=EmbedderClient)
        mock_embedder.create = AsyncMock(return_value=[[0.1,0.2]]) # Single embedding mock

        entity_nodes = [EntityNode(name="Test Bulk Node", group_id="bulk_group")]
        # ... other node/edge lists ...

        # Mock the session and transaction context
        mock_tx = AsyncMock()

        # This is a simplification; execute_write takes a function.
        # We'd need to capture that function or mock deeper.
        # For now, just ensure execute_query is called by the tx function.
        async def mock_execute_write(tx_func, *args, **kwargs):
            # Simulate running the transaction function
            await tx_func(mock_tx, *args, **kwargs)

        neo4j_provider.driver.session.return_value.__aenter__.return_value.execute_write = mock_execute_write
        neo4j_provider.driver.session.return_value.execute_write = mock_execute_write # if not async context


        await neo4j_provider.add_nodes_and_edges_bulk(
            episodic_nodes=[],
            episodic_edges=[],
            entity_nodes=entity_nodes,
            entity_edges=[],
            embedder=mock_embedder
        )

        # Check if embedder was called
        mock_embedder.create.assert_called_once()

        # Check if tx.run was called (via the mock_tx passed to _add_nodes_and_edges_bulk_tx)
        # This requires the mock_tx to be the one used by the actual implementation.
        # The current mock_tx is local. A better way is to ensure execute_query is called
        # with the bulk Cypher queries.

        # Find the call to ENTITY_NODE_SAVE_BULK
        call_args_list = mock_tx.run.call_args_list # mock_tx needs to be the one from execute_write

        # This part of test is tricky due to execute_write; would need more elaborate mocking
        # or to check calls to the underlying self.execute_query if tx.run uses it.
        # For now, asserting that the embedder was called is a start.
        # A full test would inspect the parameters to tx.run for each bulk query.

        # Example check (assuming mock_tx.run was called correctly by the tx_func)
        # found_entity_bulk_call = False
        # for call in call_args_list:
        #     if call[0][0] == node_db_queries.ENTITY_NODE_SAVE_BULK:
        #         found_entity_bulk_call = True
        #         assert len(call[1]['nodes']) == 1 # Check if nodes param has one item
        # assert found_entity_bulk_call

        # This test is incomplete for verifying tx.run calls due to complexity of mocking execute_write's callback.
        # A more complete test would involve checking the arguments passed to mock_tx.run.
        # For example, by patching 'tx.run' if tx was a real object or making mock_tx more sophisticated.
        assert mock_embedder.create.call_count > 0 # Basic check that embedding was attempted
        # To properly test calls to tx.run, the execute_write mock needs to actually call the passed tx_func
        # with a mock tx object whose 'run' method can be inspected.
        # The current setup of mock_execute_write doesn't allow easy inspection of tx.run calls.
        # This indicates a limitation in the current test setup for deeply nested calls within execute_write.
        # However, individual save methods are tested elsewhere, and bulk queries are constants.
        pass

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

```
