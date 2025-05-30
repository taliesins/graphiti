import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone
import json # For attribute serialization checks

# Assuming kuzu module is importable in the test environment.
# If not, it might need to be mocked at a higher level for some tests.
# For now, we'll mock its classes/methods as needed.
# import kuzu

from graphiti_core.providers.kuzudb_provider import KuzuDBProvider, SearchFilters
from graphiti_core.nodes import EntityNode, EpisodicNode, CommunityNode, EpisodeType
from graphiti_core.edges import EntityEdge, EpisodicEdge, CommunityEdge
from graphiti_core.embedder import EmbedderClient

# Mock data
MOCK_UUID = "mock-uuid-kuzu-123"
MOCK_GROUP_ID = "mock-group-id-kuzu"
MOCK_DATETIME_NATIVE = datetime.now(timezone.utc)


# Mock Kuzu QueryResult
class MockKuzuQueryResult:
    def __init__(self, records: List[List[Any]], column_names: List[str]):
        self._records = records
        self._column_names = column_names
        self._current_idx = 0

    def get_column_names(self) -> List[str]:
        return self._column_names

    def has_next(self) -> bool:
        return self._current_idx < len(self._records)

    def get_next(self) -> List[Any]: # Kuzu might use get_next() or iterate
        if self.has_next():
            record = self._records[self._current_idx]
            self._current_idx += 1
            return record
        raise StopIteration

    def get_as_list(self) -> List[List[Any]]: # Common way to get all results
        return self._records

    def get_as_df(self): # Mocked if needed
        # For simplicity, not implementing a full pandas DataFrame conversion here
        # Tests that rely on get_as_df would need a more sophisticated mock or pandas itself.
        # KuzuDBProvider's execute_query has a complex fallback for results, we'll mock get_as_list primarily.
        raise NotImplementedError("get_as_df mock not implemented for this test QueryResult")


@pytest_asyncio.fixture
async def mock_kuzu_connection():
    # Mock the kuzu.Connection object
    mock_conn = AsyncMock() # Use AsyncMock if connection methods are async, else MagicMock

    # Mock prepare method to return a statement object that can be executed
    mock_prepared_statement = MagicMock()
    mock_conn.prepare = MagicMock(return_value=mock_prepared_statement)

    # Mock execute method on the prepared statement or connection
    # Default: return an empty QueryResult
    empty_query_result = MockKuzuQueryResult([], [])
    # If execute is on prepared_statement:
    mock_prepared_statement.execute = MagicMock(return_value=empty_query_result)
    # If execute is directly on connection (some Kuzu versions might differ for convenience):
    mock_conn.execute = MagicMock(return_value=empty_query_result) # Fallback if prepare isn't used for some reason

    # Mock for `connection.execute("RETURN 1")` in verify_connectivity
    # This specific call might bypass prepare, so mock it directly on connection.execute
    # if it doesn't use prepare.
    # Let's assume verify_connectivity uses prepare->execute path.

    # For `build_indices_and_constraints` which calls conn.execute directly
    # for DDL statements (CREATE TABLE, DROP TABLE)
    # These typically don't return a complex QueryResult, or Kuzu handles it gracefully.
    # We can make the default mock_conn.execute more specific if needed per test.

    yield mock_conn


@pytest_asyncio.fixture
async def kuzu_provider(mock_kuzu_connection: MagicMock): # Changed to MagicMock as Kuzu conn is sync
    with patch("kuzu.Database") as mock_kuzu_db_class:
        mock_db_instance = MagicMock()
        mock_kuzu_db_class.return_value = mock_db_instance

        # Patch kuzu.Connection to return our mock_kuzu_connection
        with patch("kuzu.Connection", return_value=mock_kuzu_connection) as mock_kuzu_conn_class:
            provider = KuzuDBProvider(database_path=":memory:", in_memory=True)
            # The provider's __init__ calls _connect_internal, which creates Database and Connection.
            # So, provider.connection will be our mock_kuzu_connection.
            provider.db = mock_db_instance # Ensure db is also mocked
            provider.connection = mock_kuzu_connection
            yield provider


@pytest.mark.asyncio
class TestKuzuDBProvider:

    async def test_connect_and_close(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        # _connect_internal is called by __init__. We can test connect() for re-init logic.
        # For this test, primarily ensure close doesn't error and state is updated.
        # verify_connectivity is part of connect, so its mock on mock_kuzu_connection will be used.

        # Mock the execute call for verify_connectivity
        # Assume verify_connectivity uses prepare -> execute
        verify_stmt = MagicMock()
        mock_kuzu_connection.prepare.return_value = verify_stmt
        verify_stmt.execute = MagicMock(return_value=MockKuzuQueryResult([[]], [])) # Minimal valid result

        await kuzu_provider.connect() # Should re-verify or establish
        mock_kuzu_connection.prepare.assert_called_with("RETURN 1")
        verify_stmt.execute.assert_called_once()

        await kuzu_provider.close()
        assert kuzu_provider.connection is None
        assert kuzu_provider.db is None

    async def test_verify_connectivity_failure(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        # Make prepare or execute raise an error
        mock_prepared_stmt = MagicMock()
        mock_prepared_stmt.execute = MagicMock(side_effect=RuntimeError("Kuzu connection error"))
        mock_kuzu_connection.prepare = MagicMock(return_value=mock_prepared_stmt)

        with pytest.raises(ConnectionError, match="KuzuDB connectivity verification failed"):
            await kuzu_provider.verify_connectivity()

    async def test_save_entity_node_merge(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        node = EntityNode(
            uuid="test-uuid", name="Test Node", group_id="group1",
            summary="A test node.", attributes={"custom_attr": "val1"},
            name_embedding=[0.1,0.2]
        )
        # Expected node_dict that _save_node_generic receives
        expected_node_dict = node.model_dump(exclude_none=True)
        expected_node_dict["attributes"] = json.dumps(node.attributes)

        # Mock for the MERGE query
        merge_stmt = MagicMock()
        mock_kuzu_connection.prepare.return_value = merge_stmt
        merge_stmt.execute = MagicMock(return_value=MockKuzuQueryResult([], []))

        await kuzu_provider.save_entity_node(node)

        mock_kuzu_connection.prepare.assert_called_once()
        args, _ = mock_kuzu_connection.prepare.call_args
        query_str = args[0]

        assert f"MERGE (n:Entity {{uuid: $match_uuid}})" in query_str
        # Check for ON CREATE SET clauses (all properties from node_dict)
        for key in expected_node_dict:
            assert f"n.{key} = $p_{key}" in query_str
        # Check for ON MATCH SET clauses (all except uuid)
        for key in expected_node_dict:
            if key != "uuid":
                assert f"n.{key} = $p_{key}" in query_str

        # Check parameters passed to execute
        merge_stmt.execute.assert_called_once()
        _, execute_kwargs = merge_stmt.execute.call_args

        assert execute_kwargs["match_uuid"] == node.uuid
        for key, value in expected_node_dict.items():
            assert execute_kwargs[f"p_{key}"] == value


    async def test_get_entity_node_by_uuid_found(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        mock_node_data = {
            "uuid": MOCK_UUID, "name": "Kuzu Node", "group_id": MOCK_GROUP_ID,
            "created_at": MOCK_DATETIME_NATIVE, # Kuzu driver returns Python datetime
            "summary": "Kuzu Summary",
            "attributes": json.dumps({"prop_k": "val_k"}), # Stored as JSON string
            "name_embedding": [0.5, 0.6]
        }

        # Mock prepare and execute
        stmt_mock = MagicMock()
        stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult(
            records=[list(mock_node_data.values())], # Kuzu records are lists of values
            column_names=list(mock_node_data.keys())
        ))
        mock_kuzu_connection.prepare = MagicMock(return_value=stmt_mock)

        found_node = await kuzu_provider.get_entity_node_by_uuid(MOCK_UUID)

        assert found_node is not None
        assert found_node.uuid == MOCK_UUID
        assert found_node.name == "Kuzu Node"
        assert found_node.attributes == {"prop_k": "val_k"} # Parsed from JSON
        assert found_node.name_embedding == [0.5, 0.6]

        args, _ = mock_kuzu_connection.prepare.call_args
        query_str = args[0]
        expected_cols = "n.uuid, n.name, n.group_id, n.created_at, n.summary, n.name_embedding, n.attributes"
        assert f"MATCH (n:Entity {{uuid: $uuid}}) RETURN {expected_cols}" in query_str

        stmt_mock.execute.assert_called_once_with(uuid=MOCK_UUID)

    async def test_get_entity_node_by_uuid_not_found(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        stmt_mock = MagicMock()
        stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult([], [])) # No records
        mock_kuzu_connection.prepare = MagicMock(return_value=stmt_mock)

        found_node = await kuzu_provider.get_entity_node_by_uuid("non-existent-uuid")
        assert found_node is None

    async def test_build_indices_and_constraints_create(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        # build_indices_and_constraints uses direct connection.execute for DDL
        # We need to ensure these direct calls are captured if they don't use prepare.
        # The mock_kuzu_connection.execute is already a MagicMock.

        await kuzu_provider.build_indices_and_constraints(delete_existing=False)

        # Check some CREATE TABLE calls
        create_table_calls = [
            call_args[0][0] for call_args in mock_kuzu_connection.execute.call_args_list
            if "CREATE NODE TABLE Entity" in call_args[0][0] or \
               "CREATE NODE TABLE Episodic" in call_args[0][0] or \
               "CREATE NODE TABLE Community" in call_args[0][0] or \
               "CREATE REL TABLE MENTIONS" in call_args[0][0] or \
               "CREATE REL TABLE RELATES_TO" in call_args[0][0] or \
               "CREATE REL TABLE HAS_MEMBER" in call_args[0][0]
        ]
        assert len(create_table_calls) == 6 # 3 node tables, 3 rel tables
        # Check embedding dimension usage in Entity table schema
        assert f"FLOAT[{kuzu_provider.embedding_dimension}]" in create_table_calls[0] # Entity schema
        assert f"FLOAT[{kuzu_provider.embedding_dimension}]" in create_table_calls[2] # Community schema
        assert f"FLOAT[{kuzu_provider.embedding_dimension}]" in create_table_calls[4] # RELATES_TO schema

    async def test_build_indices_and_constraints_delete_existing(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        # Mock for direct execute calls
        # We need to track calls to connection.execute
        # If it fails on DROP (e.g. table not found), it should be caught.
        # For this test, assume DROP calls succeed or fail silently as per implementation.

        await kuzu_provider.build_indices_and_constraints(delete_existing=True)

        drop_table_calls = [
            call_args[0][0] for call_args in mock_kuzu_connection.execute.call_args_list
            if "DROP TABLE" in call_args[0][0]
        ]
        # Should attempt to drop all 6 tables
        assert len(drop_table_calls) == 6

        create_table_calls = [
            call_args[0][0] for call_args in mock_kuzu_connection.execute.call_args_list
            if "CREATE" in call_args[0][0] # Includes NODE and REL tables
        ]
        assert len(create_table_calls) == 6


    # More tests to be added for other CRUD, search, bulk operations.
    # Test for add_nodes_and_edges_bulk (conceptual)
    @patch('pandas.DataFrame') # Mock pandas DataFrame
    @patch('tempfile.NamedTemporaryFile') # Mock tempfile
    @patch('os.remove') # Mock os.remove
    async def test_add_nodes_and_edges_bulk_copy_from(
        self, mock_os_remove, mock_tempfile, mock_pd_dataframe,
        kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock
    ):
        mock_embedder = AsyncMock(spec=EmbedderClient)
        mock_embedder.create = AsyncMock(return_value=[[0.1,0.2]])

        # Mock for tempfile
        mock_tmp_file_obj = MagicMock()
        mock_tmp_file_obj.name = "dummy.parquet"
        mock_tempfile.return_value.__enter__.return_value = mock_tmp_file_obj

        # Mock for DataFrame.to_parquet
        mock_df_instance = mock_pd_dataframe.return_value
        mock_df_instance.to_parquet = MagicMock()
        mock_df_instance.empty = False # Ensure DataFrame is not considered empty

        entity_nodes = [EntityNode(uuid="en1", name="Bulk Entity 1", group_id="g1", summary="s1")]
        # Episodic nodes require source, source_description, content, valid_at
        episodic_nodes = [EpisodicNode(uuid="ep1", name="Bulk Episode 1", group_id="g1",
                                       source=EpisodeType.text, source_description="d", content="c",
                                       valid_at=MOCK_DATETIME_NATIVE)]

        # Mock for the COPY query execution
        copy_stmt_mock = MagicMock()
        copy_stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult([],[]))

        # This needs to handle multiple prepare calls for different COPY statements
        # or if execute_query doesn't use prepare for COPY
        # For simplicity, assume execute_query for COPY is direct or prepare works generally
        mock_kuzu_connection.prepare = MagicMock(return_value=copy_stmt_mock)
        mock_kuzu_connection.execute = MagicMock(return_value=MockKuzuQueryResult([],[])) # If COPY doesn't use prepare


        await kuzu_provider.add_nodes_and_edges_bulk(
            episodic_nodes=episodic_nodes,
            episodic_edges=[],
            entity_nodes=entity_nodes,
            entity_edges=[],
            embedder=mock_embedder
        )

        # Assert embedder was called
        assert mock_embedder.create.call_count > 0 # For entity_nodes

        # Assert DataFrame.to_parquet was called for nodes
        # One for Episodic, one for Entity
        assert mock_df_instance.to_parquet.call_count == 2

        # Assert that COPY FROM query was executed for each type of node
        # This requires inspecting calls to kuzu_provider.execute_query (which is not directly mocked here)
        # or connection.execute / connection.prepare().execute()

        # Check calls to kuzu_provider.connection.execute (assuming COPY doesn't use prepare)
        # OR check calls to prepared_statement.execute if COPY uses prepare.
        # The current _copy_from_df_to_kuzu calls self.execute_query, which uses prepare.

        copy_calls = [
            c_args[0] for c_args, _ in mock_kuzu_connection.prepare.call_args_list
            if "COPY" in c_args[0]
        ]
        assert len(copy_calls) == 2 # One for Episodic, one for Entity nodes
        assert "COPY Episodic FROM" in copy_calls[0]
        assert "ON_CONFLICT (uuid) DO UPDATE" in copy_calls[0] # Or DO NOTHING if no other props
        assert "COPY Entity FROM" in copy_calls[1]
        assert "ON_CONFLICT (uuid) DO UPDATE" in copy_calls[1]

        # Assert os.remove was called for the temp files
        assert mock_os_remove.call_count == 2 # One for Episodic, one for Entity

    async def test_save_episodic_node(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        node = EpisodicNode(
            uuid="ep-uuid-1", name="Test Episode", group_id="group1",
            source=EpisodeType.text, source_description="test source",
            content="test content", valid_at=MOCK_DATETIME_NATIVE,
            entity_edges=["edge-uuid-1"]
        )
        expected_node_dict = node.model_dump(exclude_none=True)
        expected_node_dict["source"] = node.source.value # Enum to value

        merge_stmt = MagicMock()
        mock_kuzu_connection.prepare.return_value = merge_stmt
        merge_stmt.execute = MagicMock(return_value=MockKuzuQueryResult([], []))

        await kuzu_provider.save_episodic_node(node)

        mock_kuzu_connection.prepare.assert_called_once()
        args, _ = mock_kuzu_connection.prepare.call_args
        query_str = args[0]
        assert f"MERGE (n:Episodic {{uuid: $match_uuid}})" in query_str
        for key in expected_node_dict:
            assert f"n.{key} = $p_{key}" in query_str

        merge_stmt.execute.assert_called_once()
        _, execute_kwargs = merge_stmt.execute.call_args
        assert execute_kwargs["match_uuid"] == node.uuid
        for key, value in expected_node_dict.items():
            assert execute_kwargs[f"p_{key}"] == value

    async def test_get_episodic_node_by_uuid_found(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        mock_ep_data = {
            "uuid": "ep-uuid-1", "name": "Kuzu Episode", "group_id": MOCK_GROUP_ID,
            "source": EpisodeType.text.value, "source_description": "desc", "content": "content",
            "valid_at": MOCK_DATETIME_NATIVE, "created_at": MOCK_DATETIME_NATIVE,
            "entity_edges": ["edge1", "edge2"]
        }
        stmt_mock = MagicMock()
        stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult(
            records=[list(mock_ep_data.values())],
            column_names=list(mock_ep_data.keys())
        ))
        mock_kuzu_connection.prepare = MagicMock(return_value=stmt_mock)

        found_node = await kuzu_provider.get_episodic_node_by_uuid("ep-uuid-1")
        assert found_node is not None
        assert found_node.name == "Kuzu Episode"
        assert found_node.source == EpisodeType.text
        assert found_node.entity_edges == ["edge1", "edge2"]

    async def test_save_entity_edge_merge(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        edge = EntityEdge(
            uuid="edge-uuid-1", name="RELATES_TO", group_id="group1",
            source_node_uuid="source-uuid", target_node_uuid="target-uuid",
            fact="test fact", fact_embedding=[0.7, 0.8], created_at=MOCK_DATETIME_NATIVE,
            attributes={"edge_prop": "edge_val"}
        )
        expected_edge_dict = edge.model_dump(exclude_none=True)
        expected_edge_dict["attributes"] = json.dumps(edge.attributes)

        # Properties for the MERGE's SET clauses (excluding source/target_node_uuid)
        props_for_rel = {k: v for k, v in expected_edge_dict.items() if k not in ['source_node_uuid', 'target_node_uuid']}


        merge_stmt = MagicMock()
        mock_kuzu_connection.prepare.return_value = merge_stmt
        merge_stmt.execute = MagicMock(return_value=MockKuzuQueryResult([], []))

        await kuzu_provider.save_entity_edge(edge)

        mock_kuzu_connection.prepare.assert_called_once()
        args, _ = mock_kuzu_connection.prepare.call_args
        query_str = args[0]

        assert f"MATCH (s:Entity {{uuid: $source_uuid}}), (t:Entity {{uuid: $target_uuid}})" in query_str
        assert f"MERGE (s)-[r:RELATES_TO {{uuid: $edge_uuid_param}}]->(t)" in query_str
        for key in props_for_rel:
            assert f"r.{key} = ${key}" in query_str # Note: _save_edge_generic uses direct keys, not p_ prefix

        merge_stmt.execute.assert_called_once()
        _, execute_kwargs = merge_stmt.execute.call_args
        assert execute_kwargs["source_uuid"] == edge.source_node_uuid
        assert execute_kwargs["target_uuid"] == edge.target_node_uuid
        assert execute_kwargs["edge_uuid_param"] == edge.uuid
        for key, value in props_for_rel.items():
            assert execute_kwargs[key] == value

    async def test_get_entity_nodes_by_uuids(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        mock_uuids = ["uuid1", "uuid2"]
        mock_node_data1 = {
            "uuid": "uuid1", "name": "Node 1", "group_id": "g1", "created_at": MOCK_DATETIME_NATIVE,
            "summary": "s1", "name_embedding": [0.1], "attributes": "{}"
        }
        mock_node_data2 = {
            "uuid": "uuid2", "name": "Node 2", "group_id": "g1", "created_at": MOCK_DATETIME_NATIVE,
            "summary": "s2", "name_embedding": [0.2], "attributes": "{}"
        }
        stmt_mock = MagicMock()
        stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult(
            records=[list(mock_node_data1.values()), list(mock_node_data2.values())],
            column_names=list(mock_node_data1.keys())
        ))
        mock_kuzu_connection.prepare = MagicMock(return_value=stmt_mock)

        results = await kuzu_provider.get_entity_nodes_by_uuids(mock_uuids)

        assert len(results) == 2
        assert results[0].uuid == "uuid1"
        assert results[1].name == "Node 2"
        args, _ = mock_kuzu_connection.prepare.call_args
        assert "MATCH (n:Entity) WHERE n.uuid IN $uuids RETURN" in args[0]
        stmt_mock.execute.assert_called_once_with(uuids=mock_uuids)

    async def test_delete_node(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        test_uuid = "delete-node-kuzu"

        # Mock for the multiple execute calls in delete_node
        # It will be called for each rel table (x2 for bi-directional) and then for node table
        # For simplicity, we'll just check that prepare was called multiple times with expected queries
        delete_stmt_mock = MagicMock()
        delete_stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult([],[]))
        mock_kuzu_connection.prepare = MagicMock(return_value=delete_stmt_mock)

        await kuzu_provider.delete_node(test_uuid)

        # Expected calls: MENTIONS (x2), RELATES_TO (x2), HAS_MEMBER (x2) for rels
        # Then Entity, Episodic, Community for nodes (stops after first success or tries all)
        # At least 7 calls to prepare if node is Entity and has all rels.
        # This is hard to assert precisely without knowing which table it's in.
        # Let's check for one specific type of call.

        calls = mock_kuzu_connection.prepare.call_args_list

        # Check if a relationship deletion query was prepared for Entity table
        assert any(f"MATCH (n:Entity {{uuid: $uuid}})-[r:MENTIONS]-() DELETE r" in call[0][0] for call in calls)
        # Check if a node deletion query was prepared
        assert any(f"MATCH (n:Entity {{uuid: $uuid}}) DELETE n" in call[0][0] for call in calls)

        # Ensure execute was called with the uuid
        # This check is simplistic as execute is called many times.
        delete_stmt_mock.execute.assert_any_call(uuid=test_uuid)

    async def test_node_similarity_search(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        mock_vector = [0.1, 0.2]
        # Using a flat structure as returned by Kuzu typically (columnar)
        mock_node_data = {
            "uuid": MOCK_UUID, "name": "Similar Node", "group_id": MOCK_GROUP_ID,
            "created_at": MOCK_DATETIME_NATIVE, "summary": "Similar Summary",
            "name_embedding": [0.1,0.2], "attributes": "{}", "score": 0.98
        }
        stmt_mock = MagicMock()
        stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult(
            records=[list(mock_node_data.values())],
            column_names=list(mock_node_data.keys())
        ))
        mock_kuzu_connection.prepare = MagicMock(return_value=stmt_mock)

        results = await kuzu_provider.node_similarity_search(search_vector=mock_vector, search_filter=SearchFilters())

        assert len(results) == 1
        assert results[0].uuid == MOCK_UUID

        args, _ = mock_kuzu_connection.prepare.call_args
        query_str = args[0]
        # FIXME: Verify KuzuDB's actual vector similarity function name and syntax.
        # This test assumes "cosine_similarity" placeholder.
        assert "cosine_similarity(n.name_embedding, $s_vec) >= $min_s" in query_str
        assert "ORDER BY score DESC" in query_str

        stmt_mock.execute.assert_called_once_with(s_vec=mock_vector, lim=10, min_s=0.6, gids=None) # gids=None if not provided

    async def test_save_community_node(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        node = CommunityNode(
            uuid="comm-uuid-1", name="Test Community", group_id="group1",
            summary="community summary", name_embedding=[0.4, 0.5]
        )
        expected_node_dict = node.model_dump(exclude_none=True)

        merge_stmt = MagicMock()
        mock_kuzu_connection.prepare.return_value = merge_stmt
        merge_stmt.execute = MagicMock(return_value=MockKuzuQueryResult([], []))

        await kuzu_provider.save_community_node(node)

        mock_kuzu_connection.prepare.assert_called_once()
        args, _ = mock_kuzu_connection.prepare.call_args
        query_str = args[0]
        assert f"MERGE (n:Community {{uuid: $match_uuid}})" in query_str
        for key in expected_node_dict:
            assert f"n.{key} = $p_{key}" in query_str

        merge_stmt.execute.assert_called_once()
        _, execute_kwargs = merge_stmt.execute.call_args
        assert execute_kwargs["match_uuid"] == node.uuid
        for key, value in expected_node_dict.items():
            assert execute_kwargs[f"p_{key}"] == value

    async def test_get_community_node_by_uuid_found(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        mock_comm_data = {
            "uuid": "comm-uuid-1", "name": "Kuzu Community", "group_id": MOCK_GROUP_ID,
            "created_at": MOCK_DATETIME_NATIVE, "summary": "Kuzu Comm Summary",
            "name_embedding": [0.7, 0.8]
        }
        stmt_mock = MagicMock()
        stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult(
            records=[list(mock_comm_data.values())],
            column_names=list(mock_comm_data.keys())
        ))
        mock_kuzu_connection.prepare = MagicMock(return_value=stmt_mock)

        found_node = await kuzu_provider.get_community_node_by_uuid("comm-uuid-1")
        assert found_node is not None
        assert found_node.name == "Kuzu Community"
        assert found_node.name_embedding == [0.7, 0.8]

    async def test_get_entity_edge_by_uuid_found(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        mock_edge_data = {
            "uuid": "edge-uuid-k1", "name": "RELATES_TO", "group_id": MOCK_GROUP_ID,
            "fact": "Kuzu fact", "fact_embedding": [0.1, 0.9], "episodes": ["ep1"],
            "created_at": MOCK_DATETIME_NATIVE, "expired_at": None, "valid_at": None, "invalid_at": None,
            "attributes": json.dumps({"edge_prop": "edge_val_k"}),
            "source_node_uuid": "source-k", "target_node_uuid": "target-k"
        }
        stmt_mock = MagicMock()
        stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult(
            records=[list(mock_edge_data.values())],
            column_names=list(mock_edge_data.keys())
        ))
        mock_kuzu_connection.prepare = MagicMock(return_value=stmt_mock)

        found_edge = await kuzu_provider.get_entity_edge_by_uuid("edge-uuid-k1")
        assert found_edge is not None
        assert found_edge.fact == "Kuzu fact"
        assert found_edge.attributes == {"edge_prop": "edge_val_k"}

        args, _ = mock_kuzu_connection.prepare.call_args
        query_str = args[0]
        assert "MATCH (s:Entity)-[r:RELATES_TO {uuid: $uuid}]->(t:Entity) RETURN" in query_str
        stmt_mock.execute.assert_called_once_with(uuid="edge-uuid-k1")


    async def test_delete_edge(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        test_uuid = "delete-edge-kuzu"
        delete_stmt_mock = MagicMock()
        delete_stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult([],[]))
        mock_kuzu_connection.prepare = MagicMock(return_value=delete_stmt_mock)

        await kuzu_provider.delete_edge(test_uuid)

        # Check that queries for all relevant rel tables were prepared
        calls = mock_kuzu_connection.prepare.call_args_list
        expected_tables = ["MENTIONS", "RELATES_TO", "HAS_MEMBER"]
        for table_name in expected_tables:
            assert any(f"MATCH ()-[r:{table_name} {{uuid: $uuid}}]-() DELETE r" in call[0][0] for call in calls)

        delete_stmt_mock.execute.assert_any_call(uuid=test_uuid)

    async def test_delete_nodes_by_group_id(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        group_id_to_delete = "group-to-delete-kuzu"
        node_type_to_delete = "Entity"

        delete_stmt_mock = MagicMock()
        delete_stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult([],[]))
        mock_kuzu_connection.prepare = MagicMock(return_value=delete_stmt_mock)

        await kuzu_provider.delete_nodes_by_group_id(group_id_to_delete, node_type_to_delete)

        calls = mock_kuzu_connection.prepare.call_args_list
        # Check for relationship deletions first
        rel_tables = ["MENTIONS", "RELATES_TO", "HAS_MEMBER"]
        for rel_table in rel_tables:
            assert any(f"MATCH (n:{node_type_to_delete} {{group_id: $gid}})-[r:{rel_table}]-() DELETE r" in call[0][0] for call in calls)
            assert any(f"MATCH (n:{node_type_to_delete} {{group_id: $gid}})<-[r:{rel_table}]-() DELETE r" in call[0][0] for call in calls)

        # Check for node deletion
        assert any(f"MATCH (n:{node_type_to_delete} {{group_id: $gid}}) DELETE n" in call[0][0] for call in calls)

        delete_stmt_mock.execute.assert_any_call(gid=group_id_to_delete)

    async def test_node_fulltext_search(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        mock_query = "search term kuzu"
        mock_node_data = {
            "uuid": MOCK_UUID, "name": "Fulltext Node", "group_id": MOCK_GROUP_ID,
            "created_at": MOCK_DATETIME_NATIVE, "summary": "A node for fulltext",
            "name_embedding": None, "attributes": "{}"
        }
        stmt_mock = MagicMock()
        stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult(
            records=[list(mock_node_data.values())],
            column_names=list(mock_node_data.keys())
        ))
        mock_kuzu_connection.prepare = MagicMock(return_value=stmt_mock)

        results = await kuzu_provider.node_fulltext_search(query=mock_query, search_filter=SearchFilters())

        assert len(results) == 1
        assert results[0].name == "Fulltext Node"

        args, _ = mock_kuzu_connection.prepare.call_args
        query_str = args[0]
        assert "CONTAINS(lower(n.name), $query_str_lower)" in query_str
        assert "CONTAINS(lower(n.summary), $query_str_lower)" in query_str
        stmt_mock.execute.assert_called_once_with(query_str_lower=mock_query.lower(), limit_val=10, gids=None)

    async def test_node_bfs_search_with_filter(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        # Assume SearchFilters is extended or _apply_search_filters_to_query_basic handles a hypothetical filter
        class ExtendedSearchFilters(SearchFilters):
            name_must_equal: Optional[str] = None

        search_filter = ExtendedSearchFilters(name_must_equal="BFS Target Node")

        mock_bfs_data = {
             "uuid": "bfs-target-uuid", "name": "BFS Target Node", "group_id": MOCK_GROUP_ID,
            "created_at": MOCK_DATETIME_NATIVE, "summary": "BFS Summary",
            "name_embedding": None, "attributes": "{}"
        }
        stmt_mock = MagicMock()
        stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult(
            records=[list(mock_bfs_data.values())],
            column_names=list(mock_bfs_data.keys())
        ))
        mock_kuzu_connection.prepare = MagicMock(return_value=stmt_mock)

        results = await kuzu_provider.node_bfs_search(
            bfs_origin_node_uuids=["origin-uuid"],
            search_filter=search_filter,
            bfs_max_depth=2,
            limit=1
        )
        assert len(results) == 1
        assert results[0].name == "BFS Target Node"

        args, _ = mock_kuzu_connection.prepare.call_args
        query_str = args[0]
        assert "MATCH (origin:Entity)-[rels*1..2 BFS]->(peer:Entity)" in query_str
        assert "peer.name = $peer_name_filter" in query_str # From _apply_search_filters_to_query_basic

        stmt_mock.execute.assert_called_once_with(
            origin_uuids=["origin-uuid"],
            lim=1,
            peer_name_filter="BFS Target Node" # Parameter added by _apply_search_filters_to_query_basic
        )

    async def test_clear_data_all(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        # build_indices_and_constraints will be called, which has its own execute calls
        # We need to ensure the sequence of calls to connection.execute is right.
        # For simplicity, we'll just check that build_indices_and_constraints was called with delete_existing=True
        # by patching it.
        with patch.object(kuzu_provider, 'build_indices_and_constraints', new_callable=AsyncMock) as mock_build:
            await kuzu_provider.clear_data(group_ids=None)
            mock_build.assert_called_once_with(delete_existing=True)

    async def test_clear_data_group_ids(self, kuzu_provider: KuzuDBProvider, mock_kuzu_connection: MagicMock):
        group_ids_to_clear = ["g1", "g2"]

        delete_stmt_mock = MagicMock()
        delete_stmt_mock.execute = MagicMock(return_value=MockKuzuQueryResult([],[]))
        mock_kuzu_connection.prepare = MagicMock(return_value=delete_stmt_mock)

        await kuzu_provider.clear_data(group_ids=group_ids_to_clear)

        # Check that various DELETE queries were prepared
        calls = mock_kuzu_connection.prepare.call_args_list
        # Example check for one type of deletion (rels with group_id)
        assert any("MATCH ()-[r:MENTIONS {group_id: $gid}]-() DELETE r" in call[0][0] for call in calls)
        # Example check for node deletion
        assert any("MATCH (n:Entity {group_id: $gid}) DELETE n" in call[0][0] for call in calls)

        # Check that execute was called with a group_id from the list
        # This is a weak check as execute is called many times.
        delete_stmt_mock.execute.assert_any_call(gid="g1")
        delete_stmt_mock.execute.assert_any_call(gid="g2")
```
