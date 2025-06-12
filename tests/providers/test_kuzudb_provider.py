import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json # For attribute serialization/deserialization
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum # Required for ComparisonOperator

# Graphiti Core Imports
from graphiti_core.providers.kuzudb_provider import KuzuDBProvider, SearchFilters, ComparisonOperator, DateFilter
from graphiti_core.nodes import EntityNode, EpisodicNode, CommunityNode, EpisodeType
from graphiti_core.edges import EntityEdge, EpisodicEdge, CommunityEdge
from graphiti_core.embedder import EmbedderClient

# Mock data
MOCK_UUID_ENTITY_1 = "mock-entity-uuid-1"
MOCK_UUID_ENTITY_2 = "mock-entity-uuid-2"
MOCK_UUID_EPISODIC_1 = "mock-episodic-uuid-1"
MOCK_UUID_EPISODIC_2 = "mock-episodic-uuid-2"
MOCK_UUID_COMMUNITY_1 = "mock-community-uuid-1"
MOCK_UUID_COMMUNITY_2 = "mock-community-uuid-2"
MOCK_UUID_EDGE_1 = "mock-edge-uuid-1" # EntityEdge
MOCK_UUID_EDGE_2 = "mock-edge-uuid-2" # EntityEdge
MOCK_UUID_EPISODIC_EDGE_1 = "mock-ep-edge-uuid-1"
MOCK_UUID_EPISODIC_EDGE_2 = "mock-ep-edge-uuid-2"
MOCK_UUID_COMMUNITY_EDGE_1 = "mock-comm-edge-uuid-1"
MOCK_UUID_COMMUNITY_EDGE_2 = "mock-comm-edge-uuid-2"


MOCK_GROUP_ID = "test-group"
MOCK_DATETIME_NATIVE = datetime.now(timezone.utc)
MOCK_FILTER_DATETIME = datetime(2023, 1, 1, tzinfo=timezone.utc)

MOCK_KUZU_ENTITY_NODE_DICT = {
    "uuid": MOCK_UUID_ENTITY_1, "name": "Test Entity Node 1", "group_id": MOCK_GROUP_ID,
    "created_at": MOCK_DATETIME_NATIVE, "summary": "Summary 1",
    "name_embedding": [0.1, 0.2, 0.3], "attributes": json.dumps({"prop1": "val1"})
}
MOCK_KUZU_ENTITY_NODE_DICT_2 = {
    "uuid": MOCK_UUID_ENTITY_2, "name": "Test Entity Node 2", "group_id": MOCK_GROUP_ID,
    "created_at": MOCK_DATETIME_NATIVE, "summary": "Summary 2",
    "name_embedding": [0.4, 0.5, 0.6], "attributes": json.dumps({"prop2": "val2"})
}
MOCK_KUZU_EPISODIC_NODE_DICT = {
    "uuid": MOCK_UUID_EPISODIC_1, "name": "Test Episodic Node 1", "group_id": MOCK_GROUP_ID,
    "source": EpisodeType.text.value, "source_description": "Source desc 1",
    "content": "Content 1", "valid_at": MOCK_DATETIME_NATIVE, "created_at": MOCK_DATETIME_NATIVE,
    "entity_edges": [MOCK_UUID_EDGE_1]
}
MOCK_KUZU_EPISODIC_NODE_DICT_2 = {
    "uuid": MOCK_UUID_EPISODIC_2, "name": "Test Episodic Node 2", "group_id": MOCK_GROUP_ID,
    "source": EpisodeType.email.value, "source_description": "Source desc 2",
    "content": "Content 2", "valid_at": MOCK_DATETIME_NATIVE, "created_at": MOCK_DATETIME_NATIVE,
    "entity_edges": [MOCK_UUID_EDGE_2]
}
MOCK_KUZU_COMMUNITY_NODE_DICT = {
    "uuid": MOCK_UUID_COMMUNITY_1, "name": "Test Community Node 1", "group_id": MOCK_GROUP_ID,
    "created_at": MOCK_DATETIME_NATIVE, "summary": "Community Summary 1",
    "name_embedding": [0.7, 0.8, 0.9]
}
MOCK_KUZU_COMMUNITY_NODE_DICT_2 = {
    "uuid": MOCK_UUID_COMMUNITY_2, "name": "Test Community Node 2", "group_id": MOCK_GROUP_ID,
    "created_at": MOCK_DATETIME_NATIVE, "summary": "Community Summary 2",
    "name_embedding": [0.11, 0.22, 0.33]
}
MOCK_KUZU_ENTITY_EDGE_DICT = {
    "uuid": MOCK_UUID_EDGE_1, "name": "RELATES_TO_TEST_1", "group_id": MOCK_GROUP_ID,
    "fact": "Fact 1.", "fact_embedding": [0.1, 0.2, 0.3], "episodes": [MOCK_UUID_EPISODIC_1],
    "created_at": MOCK_DATETIME_NATIVE, "expired_at": None, "valid_at": MOCK_DATETIME_NATIVE,
    "invalid_at": None, "attributes": json.dumps({"edge_prop1": "val1"}),
    "source_node_uuid": MOCK_UUID_ENTITY_1, "target_node_uuid": MOCK_UUID_ENTITY_2
}
MOCK_KUZU_ENTITY_EDGE_DICT_2 = {
    "uuid": MOCK_UUID_EDGE_2, "name": "RELATES_TO_TEST_2", "group_id": MOCK_GROUP_ID,
    "fact": "Fact 2.", "fact_embedding": [0.4, 0.5, 0.6], "episodes": [MOCK_UUID_EPISODIC_2],
    "created_at": MOCK_DATETIME_NATIVE, "expired_at": None, "valid_at": MOCK_DATETIME_NATIVE,
    "invalid_at": None, "attributes": json.dumps({"edge_prop2": "val2"}),
    "source_node_uuid": MOCK_UUID_ENTITY_2, "target_node_uuid": MOCK_UUID_ENTITY_1
}
MOCK_KUZU_EPISODIC_EDGE_DICT = {
    "uuid": MOCK_UUID_EPISODIC_EDGE_1, "group_id": MOCK_GROUP_ID, "created_at": MOCK_DATETIME_NATIVE,
    "source_node_uuid": MOCK_UUID_EPISODIC_1, "target_node_uuid": MOCK_UUID_ENTITY_1
}
MOCK_KUZU_EPISODIC_EDGE_DICT_2 = {
    "uuid": MOCK_UUID_EPISODIC_EDGE_2, "group_id": MOCK_GROUP_ID, "created_at": MOCK_DATETIME_NATIVE,
    "source_node_uuid": MOCK_UUID_EPISODIC_2, "target_node_uuid": MOCK_UUID_ENTITY_2
}
MOCK_KUZU_COMMUNITY_EDGE_DICT = {
    "uuid": MOCK_UUID_COMMUNITY_EDGE_1, "group_id": MOCK_GROUP_ID, "created_at": MOCK_DATETIME_NATIVE,
    "source_node_uuid": MOCK_UUID_COMMUNITY_1, "target_node_uuid": MOCK_UUID_ENTITY_1
}
MOCK_KUZU_COMMUNITY_EDGE_DICT_2 = {
    "uuid": MOCK_UUID_COMMUNITY_EDGE_2, "group_id": MOCK_GROUP_ID, "created_at": MOCK_DATETIME_NATIVE,
    "source_node_uuid": MOCK_UUID_COMMUNITY_2, "target_node_uuid": MOCK_UUID_ENTITY_2
}

EMPTY_SEARCH_FILTERS = SearchFilters()
MOCK_SEARCH_FILTERS = SearchFilters(
    created_at_filter=DateFilter(date=MOCK_FILTER_DATETIME, operator=ComparisonOperator.GREATER_THAN_OR_EQUALS),
    group_id_filter="specific-group-for-filter-test"
)

def configure_mock_query_result(mock_qr_object: MagicMock, column_names: List[str], rows: List[List[Any]], has_next: bool = True):
    mock_qr_object.has_next.return_value = has_next
    mock_qr_object.get_column_names.return_value = column_names
    mock_qr_object.get_as_list.return_value = rows
    mock_df = MagicMock(); mock_df.to_dict.return_value = {'data': rows}; mock_qr_object.get_as_df.return_value = mock_df
    mock_torch_data = MagicMock(); mock_torch_data.node_features = rows; mock_qr_object.get_as_torch_geometric.return_value = mock_torch_data

@pytest_asyncio.fixture
async def mock_kuzu_connection():
    mock_conn_instance = AsyncMock(name="mock_kuzu_actual_connection")
    mock_prepared_statement = MagicMock(name="mock_kuzu_prepared_statement")
    mock_conn_instance.prepare = MagicMock(return_value=mock_prepared_statement)
    mock_query_result = MagicMock(name="mock_kuzu_query_result")
    configure_mock_query_result(mock_query_result, [], [], has_next=False)
    mock_conn_instance.execute = MagicMock(return_value=mock_query_result)
    yield mock_conn_instance

@pytest_asyncio.fixture
async def kuzudb_provider(mock_kuzu_connection: AsyncMock):
    with patch("kuzu.Database") as mock_kuzu_db_class, patch("kuzu.Connection") as mock_kuzu_conn_class:
        mock_db_instance = MagicMock(name="mock_kuzu_db_instance")
        mock_kuzu_db_class.return_value = mock_db_instance
        mock_kuzu_conn_class.return_value = mock_kuzu_connection
        provider = KuzuDBProvider(database_path=":memory:", in_memory=True, embedding_dimension=3)
        assert provider.connection == mock_kuzu_connection
        original_prepare_side_effect = mock_kuzu_connection.prepare.side_effect
        original_execute_side_effect = mock_kuzu_connection.execute.side_effect
        original_prepare_return_value = mock_kuzu_connection.prepare.return_value
        original_execute_return_value = mock_kuzu_connection.execute.return_value
        ps_return_1 = MagicMock(name="ps_return_1_for_verify")
        qr_return_1 = MagicMock(name="qr_return_1_for_verify")
        configure_mock_query_result(qr_return_1, ["1"], [[1]], has_next=True)
        def temp_prepare_side_effect(query_str):
            if query_str == "RETURN 1": return ps_return_1
            if original_prepare_side_effect: return original_prepare_side_effect(query_str)
            return MagicMock(name=f"default_ps_in_temp_prepare_for_{query_str}")
        def temp_execute_side_effect(prepared_statement, **kwargs):
            if prepared_statement == ps_return_1: return qr_return_1
            if original_execute_side_effect: return original_execute_side_effect(prepared_statement, **kwargs)
            default_qr = MagicMock(name=f"default_qr_in_temp_execute")
            configure_mock_query_result(default_qr, [], [], has_next=False)
            return default_qr
        mock_kuzu_connection.prepare.side_effect = temp_prepare_side_effect
        mock_kuzu_connection.execute.side_effect = temp_execute_side_effect
        await provider.connect()
        mock_kuzu_connection.prepare.side_effect = original_prepare_side_effect
        mock_kuzu_connection.execute.side_effect = original_execute_side_effect
        if not original_prepare_side_effect: mock_kuzu_connection.prepare.return_value = original_prepare_return_value
        if not original_execute_side_effect: mock_kuzu_connection.execute.return_value = original_execute_return_value
        yield provider

@pytest.mark.asyncio
class TestKuzuDBProvider:
    async def test_connect_and_close(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        await kuzudb_provider.close()
        assert kuzudb_provider.connection is None and kuzudb_provider.db is None

    async def test_verify_connectivity_failure(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        original_execute = mock_kuzu_connection.execute
        mock_kuzu_connection.execute = AsyncMock(side_effect=Exception("Simulated Kuzu connection error"))
        with pytest.raises(ConnectionError, match="KuzuDB connectivity verification failed"):
            await kuzudb_provider.verify_connectivity()
        mock_kuzu_connection.execute = original_execute

    # --- Node Save/Get Tests ---
    async def test_save_entity_node(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        node = EntityNode(uuid=MOCK_UUID_ENTITY_1, name="N1", group_id=MOCK_GROUP_ID, attributes={"k": "v"}, name_embedding=[.1,.2,.3])
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps
        await kuzudb_provider.save_entity_node(node)
        mock_kuzu_connection.prepare.assert_called_once()
        actual_kwargs = mock_kuzu_connection.execute.call_args[1]
        assert actual_kwargs["p_attributes"] == json.dumps({"k": "v"})
        assert "MERGE (n:Entity {uuid: $match_uuid})" in mock_kuzu_connection.prepare.call_args[0][0]

    async def test_get_entity_node_by_uuid_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes"]
        r_vals = [MOCK_KUZU_ENTITY_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r_vals]); mock_kuzu_connection.execute.return_value = mock_qr
        node = await kuzudb_provider.get_entity_node_by_uuid(MOCK_UUID_ENTITY_1)
        assert node is not None and node.uuid == MOCK_UUID_ENTITY_1

    async def test_get_entity_node_by_uuid_not_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        configure_mock_query_result(mock_qr,[],[],has_next=False); mock_kuzu_connection.execute.return_value=mock_qr
        assert await kuzudb_provider.get_entity_node_by_uuid(" DNE") is None

    async def test_get_entity_nodes_by_uuids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        uuids = [MOCK_UUID_ENTITY_1, MOCK_UUID_ENTITY_2]
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes"]
        r1 = [MOCK_KUZU_ENTITY_NODE_DICT[k.split('.')[1]] for k in cols]
        r2 = [MOCK_KUZU_ENTITY_NODE_DICT_2[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1, r2]); mock_kuzu_connection.execute.return_value=mock_qr
        nodes = await kuzudb_provider.get_entity_nodes_by_uuids(uuids)
        assert len(nodes) == 2 and nodes[1].uuid == MOCK_UUID_ENTITY_2
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "MATCH (n:Entity) WHERE n.uuid IN $uuids RETURN" in query_str
        assert mock_kuzu_connection.execute.call_args[1]['uuids'] == uuids

    async def test_get_entity_nodes_by_uuids_some_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        uuids = [MOCK_UUID_ENTITY_1, "dne-uuid"]
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes"]
        r1 = [MOCK_KUZU_ENTITY_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1]); mock_kuzu_connection.execute.return_value=mock_qr
        nodes = await kuzudb_provider.get_entity_nodes_by_uuids(uuids)
        assert len(nodes) == 1
        assert nodes[0].uuid == MOCK_UUID_ENTITY_1

    async def test_get_entity_nodes_by_uuids_none_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        uuids = ["dne-uuid-1", "dne-uuid-2"]
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        configure_mock_query_result(mock_qr, [], [], has_next=False); mock_kuzu_connection.execute.return_value=mock_qr
        nodes = await kuzudb_provider.get_entity_nodes_by_uuids(uuids)
        assert len(nodes) == 0

    async def test_get_entity_nodes_by_uuids_empty_input(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        nodes = await kuzudb_provider.get_entity_nodes_by_uuids([])
        assert len(nodes) == 0
        mock_kuzu_connection.prepare.assert_not_called()

    async def test_get_entity_nodes_by_group_ids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes"]
        r1 = [MOCK_KUZU_ENTITY_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1]); mock_kuzu_connection.execute.return_value=mock_qr
        nodes = await kuzudb_provider.get_entity_nodes_by_group_ids([MOCK_GROUP_ID], limit=1)
        assert len(nodes) == 1 and nodes[0].uuid == MOCK_UUID_ENTITY_1
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "MATCH (n:Entity) WHERE n.group_id IN $group_ids RETURN" in query_str
        assert "LIMIT $limit" in query_str
        assert mock_kuzu_connection.execute.call_args[1]['group_ids'] == [MOCK_GROUP_ID]
        assert mock_kuzu_connection.execute.call_args[1]['limit'] == 1


    async def test_get_entity_nodes_by_group_ids_no_limit(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes"]
        r1 = [MOCK_KUZU_ENTITY_NODE_DICT[k.split('.')[1]] for k in cols]; r2 = [MOCK_KUZU_ENTITY_NODE_DICT_2[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1, r2]); mock_kuzu_connection.execute.return_value=mock_qr
        nodes = await kuzudb_provider.get_entity_nodes_by_group_ids([MOCK_GROUP_ID])
        assert len(nodes) == 2
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "LIMIT $limit" not in query_str

    async def test_get_entity_nodes_by_group_ids_no_results(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        configure_mock_query_result(mock_qr, [], [], has_next=False); mock_kuzu_connection.execute.return_value=mock_qr
        nodes = await kuzudb_provider.get_entity_nodes_by_group_ids(["non-existent-group"], limit=5)
        assert len(nodes) == 0

    async def test_get_entity_nodes_by_group_ids_empty_input(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        nodes = await kuzudb_provider.get_entity_nodes_by_group_ids([], limit=5)
        assert len(nodes) == 0
        mock_kuzu_connection.prepare.assert_not_called()


    # (Episodic & Community Node Save/Get/GetByGroup omitted for brevity but follow EntityNode structure)
    async def test_save_episodic_node(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        node = EpisodicNode(uuid=MOCK_UUID_EPISODIC_1, name="N1", group_id=MOCK_GROUP_ID, source=EpisodeType.text, entity_edges=[])
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock(); configure_mock_query_result(mock_qr,[],[], has_next=False); mock_kuzu_connection.execute.return_value=mock_qr
        await kuzudb_provider.save_episodic_node(node)
        assert "MERGE (n:Episodic {uuid: $match_uuid})" in mock_kuzu_connection.prepare.call_args[0][0]
        assert mock_kuzu_connection.execute.call_args[1]['p_source'] == EpisodeType.text.value

    async def test_get_episodic_node_by_uuid_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.source", "n.source_description", "n.content", "n.valid_at", "n.created_at", "n.entity_edges"]
        r_vals = [MOCK_KUZU_EPISODIC_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r_vals]); mock_kuzu_connection.execute.return_value = mock_qr
        node = await kuzudb_provider.get_episodic_node_by_uuid(MOCK_UUID_EPISODIC_1)
        assert node is not None and node.uuid == MOCK_UUID_EPISODIC_1
        assert node.source == EpisodeType(MOCK_KUZU_EPISODIC_NODE_DICT["source"])

    async def test_get_episodic_node_by_uuid_not_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        configure_mock_query_result(mock_qr,[],[],has_next=False); mock_kuzu_connection.execute.return_value=mock_qr
        assert await kuzudb_provider.get_episodic_node_by_uuid("DNE") is None

    async def test_get_episodic_nodes_by_uuids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        uuids = [MOCK_UUID_EPISODIC_1, MOCK_UUID_EPISODIC_2]
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.source", "n.source_description", "n.content", "n.valid_at", "n.created_at", "n.entity_edges"]
        r1 = [MOCK_KUZU_EPISODIC_NODE_DICT[k.split('.')[1]] for k in cols]
        r2 = [MOCK_KUZU_EPISODIC_NODE_DICT_2[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1, r2]); mock_kuzu_connection.execute.return_value=mock_qr
        nodes = await kuzudb_provider.get_episodic_nodes_by_uuids(uuids)
        assert len(nodes) == 2 and nodes[1].uuid == MOCK_UUID_EPISODIC_2

    async def test_get_episodic_nodes_by_group_ids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.source", "n.source_description", "n.content", "n.valid_at", "n.created_at", "n.entity_edges"]
        r1 = [MOCK_KUZU_EPISODIC_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1]); mock_kuzu_connection.execute.return_value=mock_qr
        nodes = await kuzudb_provider.get_episodic_nodes_by_group_ids([MOCK_GROUP_ID], limit=1)
        assert len(nodes) == 1 and nodes[0].uuid == MOCK_UUID_EPISODIC_1

    async def test_save_community_node(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        node = CommunityNode(uuid=MOCK_UUID_COMMUNITY_1, name="N1", group_id=MOCK_GROUP_ID, name_embedding=[.1,.2,.3])
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock(); configure_mock_query_result(mock_qr,[],[], has_next=False); mock_kuzu_connection.execute.return_value=mock_qr
        await kuzudb_provider.save_community_node(node)
        assert "MERGE (n:Community {uuid: $match_uuid})" in mock_kuzu_connection.prepare.call_args[0][0]

    async def test_get_community_node_by_uuid_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding"]
        r_vals = [MOCK_KUZU_COMMUNITY_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r_vals]); mock_kuzu_connection.execute.return_value = mock_qr
        node = await kuzudb_provider.get_community_node_by_uuid(MOCK_UUID_COMMUNITY_1)
        assert node is not None and node.uuid == MOCK_UUID_COMMUNITY_1
        assert node.name_embedding == MOCK_KUZU_COMMUNITY_NODE_DICT["name_embedding"]

    async def test_get_community_node_by_uuid_not_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        configure_mock_query_result(mock_qr,[],[],has_next=False); mock_kuzu_connection.execute.return_value=mock_qr
        assert await kuzudb_provider.get_community_node_by_uuid("DNE") is None

    async def test_get_community_nodes_by_uuids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        uuids = [MOCK_UUID_COMMUNITY_1, MOCK_UUID_COMMUNITY_2]
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding"]
        r1 = [MOCK_KUZU_COMMUNITY_NODE_DICT[k.split('.')[1]] for k in cols]
        r2 = [MOCK_KUZU_COMMUNITY_NODE_DICT_2[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1, r2]); mock_kuzu_connection.execute.return_value=mock_qr
        nodes = await kuzudb_provider.get_community_nodes_by_uuids(uuids)
        assert len(nodes) == 2 and nodes[1].uuid == MOCK_UUID_COMMUNITY_2

    async def test_get_community_nodes_by_group_ids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding"]
        r1 = [MOCK_KUZU_COMMUNITY_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1]); mock_kuzu_connection.execute.return_value=mock_qr
        nodes = await kuzudb_provider.get_community_nodes_by_group_ids([MOCK_GROUP_ID], limit=1)
        assert len(nodes) == 1 and nodes[0].uuid == MOCK_UUID_COMMUNITY_1

    # --- Edge Save/Get Tests ---
    async def test_save_entity_edge(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        edge = EntityEdge(uuid=MOCK_UUID_EDGE_1, name="R1", group_id=MOCK_GROUP_ID, source_node_uuid=MOCK_UUID_ENTITY_1, target_node_uuid=MOCK_UUID_ENTITY_2, fact="fact")
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock(); configure_mock_query_result(mock_qr,[],[], has_next=False); mock_kuzu_connection.execute.return_value=mock_qr
        await kuzudb_provider.save_entity_edge(edge)
        assert "MERGE (s:Entity {uuid: $source_uuid})-[r:RELATES_TO {uuid: $edge_uuid_param}]->(t:Entity {uuid: $target_uuid})" in mock_kuzu_connection.prepare.call_args[0][0]
        actual_kwargs = mock_kuzu_connection.execute.call_args[1]
        assert actual_kwargs['p_attributes'] == json.dumps({}) # Default from EntityEdge if not set

    async def test_get_entity_edge_by_uuid_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["r.uuid", "r.name", "r.group_id", "r.fact", "r.fact_embedding", "r.episodes", "r.created_at", "r.expired_at", "r.valid_at", "r.invalid_at", "r.attributes", "source_node_uuid", "target_node_uuid"]
        data_dict = MOCK_KUZU_ENTITY_EDGE_DICT
        row_vals = [data_dict[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr, cols, [row_vals]); mock_kuzu_connection.execute.return_value=mock_qr
        edge = await kuzudb_provider.get_entity_edge_by_uuid(MOCK_UUID_EDGE_1)
        assert edge is not None and edge.uuid == MOCK_UUID_EDGE_1
        assert edge.attributes == json.loads(MOCK_KUZU_ENTITY_EDGE_DICT["attributes"])

    async def test_get_entity_edge_by_uuid_not_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        configure_mock_query_result(mock_qr,[],[],has_next=False); mock_kuzu_connection.execute.return_value=mock_qr
        assert await kuzudb_provider.get_entity_edge_by_uuid("DNE") is None

    async def test_get_entity_edges_by_uuids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        uuids = [MOCK_UUID_EDGE_1, MOCK_UUID_EDGE_2]
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["r.uuid", "r.name", "r.group_id", "r.fact", "r.fact_embedding", "r.episodes", "r.created_at", "r.expired_at", "r.valid_at", "r.invalid_at", "r.attributes", "source_node_uuid", "target_node_uuid"]
        r1 = [MOCK_KUZU_ENTITY_EDGE_DICT[k.split('.')[1] if '.' in k else k] for k in cols]
        r2 = [MOCK_KUZU_ENTITY_EDGE_DICT_2[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1, r2]); mock_kuzu_connection.execute.return_value=mock_qr
        edges = await kuzudb_provider.get_entity_edges_by_uuids(uuids)
        assert len(edges) == 2 and edges[1].uuid == MOCK_UUID_EDGE_2

    async def test_get_entity_edges_by_group_ids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["r.uuid", "r.name", "r.group_id", "r.fact", "r.fact_embedding", "r.episodes", "r.created_at", "r.expired_at", "r.valid_at", "r.invalid_at", "r.attributes", "source_node_uuid", "target_node_uuid"]
        r1 = [MOCK_KUZU_ENTITY_EDGE_DICT[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1]); mock_kuzu_connection.execute.return_value=mock_qr
        edges = await kuzudb_provider.get_entity_edges_by_group_ids([MOCK_GROUP_ID], limit=1)
        assert len(edges) == 1 and edges[0].uuid == MOCK_UUID_EDGE_1
        assert "LIMIT $limit" in mock_kuzu_connection.prepare.call_args[0][0]

    # (Episodic & Community Edge Save/Get/GetByGroup omitted for brevity)
    async def test_save_episodic_edge(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        edge = EpisodicEdge(uuid=MOCK_UUID_EPISODIC_EDGE_1, group_id=MOCK_GROUP_ID, source_node_uuid=MOCK_UUID_EPISODIC_1, target_node_uuid=MOCK_UUID_ENTITY_1)
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock(); configure_mock_query_result(mock_qr,[],[], has_next=False); mock_kuzu_connection.execute.return_value=mock_qr
        await kuzudb_provider.save_episodic_edge(edge)
        assert "MERGE (s:Episodic {uuid: $source_uuid})-[r:MENTIONS {uuid: $edge_uuid_param}]->(t:Entity {uuid: $target_uuid})" in mock_kuzu_connection.prepare.call_args[0][0]

    async def test_get_episodic_edge_by_uuid_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["r.uuid", "r.group_id", "r.created_at", "source_node_uuid", "target_node_uuid"]
        r_vals = [MOCK_KUZU_EPISODIC_EDGE_DICT[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r_vals]); mock_kuzu_connection.execute.return_value=mock_qr
        edge = await kuzudb_provider.get_episodic_edge_by_uuid(MOCK_UUID_EPISODIC_EDGE_1)
        assert edge is not None and edge.uuid == MOCK_UUID_EPISODIC_EDGE_1

    async def test_get_episodic_edges_by_uuids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        uuids = [MOCK_UUID_EPISODIC_EDGE_1, MOCK_UUID_EPISODIC_EDGE_2]
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["r.uuid", "r.group_id", "r.created_at", "source_node_uuid", "target_node_uuid"]
        r1 = [MOCK_KUZU_EPISODIC_EDGE_DICT[k.split('.')[1] if '.' in k else k] for k in cols]
        r2 = [MOCK_KUZU_EPISODIC_EDGE_DICT_2[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1, r2]); mock_kuzu_connection.execute.return_value=mock_qr
        edges = await kuzudb_provider.get_episodic_edges_by_uuids(uuids)
        assert len(edges) == 2 and edges[1].uuid == MOCK_UUID_EPISODIC_EDGE_2

    async def test_get_episodic_edges_by_group_ids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["r.uuid", "r.group_id", "r.created_at", "source_node_uuid", "target_node_uuid"]
        r1 = [MOCK_KUZU_EPISODIC_EDGE_DICT[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1]); mock_kuzu_connection.execute.return_value=mock_qr
        edges = await kuzudb_provider.get_episodic_edges_by_group_ids([MOCK_GROUP_ID], limit=1)
        assert len(edges) == 1 and edges[0].uuid == MOCK_UUID_EPISODIC_EDGE_1

    async def test_save_community_edge(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        edge = CommunityEdge(uuid=MOCK_UUID_COMMUNITY_EDGE_1, group_id=MOCK_GROUP_ID, source_node_uuid=MOCK_UUID_COMMUNITY_1, target_node_uuid=MOCK_UUID_ENTITY_1)
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock(); configure_mock_query_result(mock_qr,[],[], has_next=False); mock_kuzu_connection.execute.return_value=mock_qr
        await kuzudb_provider.save_community_edge(edge)
        assert "MERGE (s:Community {uuid: $source_uuid})-[r:HAS_MEMBER {uuid: $edge_uuid_param}]->(t:Entity {uuid: $target_uuid})" in mock_kuzu_connection.prepare.call_args[0][0]

    async def test_get_community_edge_by_uuid_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["r.uuid", "r.group_id", "r.created_at", "source_node_uuid", "target_node_uuid"]
        r_vals = [MOCK_KUZU_COMMUNITY_EDGE_DICT[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r_vals]); mock_kuzu_connection.execute.return_value=mock_qr
        edge = await kuzudb_provider.get_community_edge_by_uuid(MOCK_UUID_COMMUNITY_EDGE_1)
        assert edge is not None and edge.uuid == MOCK_UUID_COMMUNITY_EDGE_1

    async def test_get_community_edges_by_uuids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        uuids = [MOCK_UUID_COMMUNITY_EDGE_1, MOCK_UUID_COMMUNITY_EDGE_2]
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["r.uuid", "r.group_id", "r.created_at", "source_node_uuid", "target_node_uuid"]
        r1 = [MOCK_KUZU_COMMUNITY_EDGE_DICT[k.split('.')[1] if '.' in k else k] for k in cols]
        r2 = [MOCK_KUZU_COMMUNITY_EDGE_DICT_2[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1, r2]); mock_kuzu_connection.execute.return_value=mock_qr
        edges = await kuzudb_provider.get_community_edges_by_uuids(uuids)
        assert len(edges) == 2 and edges[1].uuid == MOCK_UUID_COMMUNITY_EDGE_2

    async def test_get_community_edges_by_group_ids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["r.uuid", "r.group_id", "r.created_at", "source_node_uuid", "target_node_uuid"]
        r1 = [MOCK_KUZU_COMMUNITY_EDGE_DICT[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1]); mock_kuzu_connection.execute.return_value=mock_qr
        edges = await kuzudb_provider.get_community_edges_by_group_ids([MOCK_GROUP_ID], limit=1)
        assert len(edges) == 1 and edges[0].uuid == MOCK_UUID_COMMUNITY_EDGE_1

    # --- Search Method Tests ---
    async def test_node_fulltext_search(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_query = "Test"; mock_limit = 5
        mock_ps = MagicMock(name="ps_node_fts"); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock(name="qr_node_fts")
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes"]
        row1 = [MOCK_KUZU_ENTITY_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [row1]); mock_kuzu_connection.execute.return_value = mock_qr
        results = await kuzudb_provider.node_fulltext_search(query=search_query, search_filter=EMPTY_SEARCH_FILTERS, group_ids=[MOCK_GROUP_ID], limit=mock_limit)
        assert len(results) == 1 and results[0].uuid == MOCK_KUZU_ENTITY_NODE_DICT["uuid"]
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "CONTAINS(lower(n.name), $query_str_lower)" in query_str and "n.group_id IN $gids" in query_str
        expected_params = {'query_str_lower': search_query.lower(), 'limit_val': mock_limit, 'gids': [MOCK_GROUP_ID]}
        mock_kuzu_connection.execute.assert_called_once_with(mock_ps, **expected_params)

    async def test_node_fulltext_search_with_search_filters(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_query = "Test"; mock_limit = 5
        mock_ps = MagicMock(name="ps_node_fts_sf"); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock(name="qr_node_fts_sf")
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes"]
        row1 = [MOCK_KUZU_ENTITY_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [row1]); mock_kuzu_connection.execute.return_value = mock_qr

        results = await kuzudb_provider.node_fulltext_search(query=search_query, search_filter=MOCK_SEARCH_FILTERS, group_ids=[MOCK_GROUP_ID], limit=mock_limit)
        assert len(results) == 1
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "n.created_at >= $n_created_at_date" in query_str # From MOCK_SEARCH_FILTERS
        assert "n.group_id = $n_group_id_filter" in query_str # From MOCK_SEARCH_FILTERS
        actual_params = mock_kuzu_connection.execute.call_args[1]
        assert actual_params['n_created_at_date'] == MOCK_FILTER_DATETIME
        assert actual_params['n_group_id_filter'] == "specific-group-for-filter-test"

    async def test_node_similarity_search(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_vector = [0.1,0.2,0.3]; mock_limit=3; min_score=0.75
        mock_ps=MagicMock(name="ps_node_sim");mock_kuzu_connection.prepare.return_value=mock_ps;mock_qr=MagicMock(name="qr_node_sim")
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes", "score"]
        row1 = [MOCK_KUZU_ENTITY_NODE_DICT[k.split('.')[1] if k != "score" else "uuid"] for k in cols]; row1[-1]=0.8 # score
        configure_mock_query_result(mock_qr,cols,[row1]);mock_kuzu_connection.execute.return_value=mock_qr
        results = await kuzudb_provider.node_similarity_search(search_vector=search_vector, search_filter=EMPTY_SEARCH_FILTERS, group_ids=[MOCK_GROUP_ID], limit=mock_limit, min_score=min_score)
        assert len(results)==1
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "cosine_similarity(n.name_embedding, $s_vec) >= $min_s" in query_str

    async def test_edge_fulltext_search(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_query="Test fact"; mock_limit=2
        mock_ps=MagicMock(name="ps_edge_fts");mock_kuzu_connection.prepare.return_value=mock_ps;mock_qr=MagicMock(name="qr_edge_fts")
        cols = ["r.uuid", "r.name", "r.group_id", "r.fact", "r.fact_embedding", "r.episodes", "r.created_at", "r.expired_at", "r.valid_at", "r.invalid_at", "r.attributes", "source_node_uuid", "target_node_uuid"]
        row1 = [MOCK_KUZU_ENTITY_EDGE_DICT[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr,cols,[row1]);mock_kuzu_connection.execute.return_value=mock_qr
        results = await kuzudb_provider.edge_fulltext_search(query=search_query, search_filter=EMPTY_SEARCH_FILTERS, group_ids=[MOCK_GROUP_ID], limit=mock_limit)
        assert len(results)==1 and results[0].uuid == MOCK_UUID_EDGE_1
        assert "MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)" in mock_kuzu_connection.prepare.call_args[0][0]

    async def test_edge_similarity_search(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_vector=[.1,.2,.3]; mock_limit=1; min_score=0.8
        mock_ps=MagicMock();mock_kuzu_connection.prepare.return_value=mock_ps;mock_qr=MagicMock()
        cols = ["r.uuid", "r.name", "r.group_id", "r.fact", "r.fact_embedding", "r.episodes", "r.created_at", "r.expired_at", "r.valid_at", "r.invalid_at", "r.attributes", "source_node_uuid", "target_node_uuid", "score"]
        row1 = [MOCK_KUZU_ENTITY_EDGE_DICT[k.split('.')[1] if k != "score" and '.' in k else k] for k in cols]; row1[-1]=0.9
        configure_mock_query_result(mock_qr,cols,[row1]);mock_kuzu_connection.execute.return_value=mock_qr
        results = await kuzudb_provider.edge_similarity_search(search_vector=search_vector, search_filter=EMPTY_SEARCH_FILTERS, group_ids=[MOCK_GROUP_ID], limit=mock_limit, min_score=min_score)
        assert len(results)==1
        assert "cosine_similarity(r.fact_embedding, $s_vec) >= $min_s" in mock_kuzu_connection.prepare.call_args[0][0]

    async def test_episode_fulltext_search(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_query="Content"; mock_limit=1
        mock_ps=MagicMock();mock_kuzu_connection.prepare.return_value=mock_ps;mock_qr=MagicMock()
        cols = ["e.uuid", "e.name", "e.group_id", "e.source", "e.source_description", "e.content", "e.valid_at", "e.created_at", "e.entity_edges"]
        row1 = [MOCK_KUZU_EPISODIC_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr,cols,[row1]);mock_kuzu_connection.execute.return_value=mock_qr
        results = await kuzudb_provider.episode_fulltext_search(query=search_query, search_filter=EMPTY_SEARCH_FILTERS, group_ids=[MOCK_GROUP_ID], limit=mock_limit)
        assert len(results)==1 and results[0].uuid == MOCK_UUID_EPISODIC_1
        assert "MATCH (e:Episodic)" in mock_kuzu_connection.prepare.call_args[0][0]
        assert "CONTAINS(lower(e.content), $query_str_lower)" in mock_kuzu_connection.prepare.call_args[0][0]

    async def test_community_fulltext_search(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_query="Summary"; mock_limit=1
        s_filters = SearchFilters(created_at_filter=DateFilter(date=MOCK_FILTER_DATETIME, operator=ComparisonOperator.EQUALS))

        mock_ps=MagicMock();mock_kuzu_connection.prepare.return_value=mock_ps;mock_qr=MagicMock()
        cols = ["c.uuid", "c.name", "c.group_id", "c.created_at", "c.summary", "c.name_embedding"]
        # Adjust mock data to satisfy the new filter for testing purposes
        community_dict_match = {**MOCK_KUZU_COMMUNITY_NODE_DICT, "created_at": MOCK_FILTER_DATETIME}
        row1 = [community_dict_match[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr,cols,[row1]);mock_kuzu_connection.execute.return_value=mock_qr

        results = await kuzudb_provider.community_fulltext_search(
            query=search_query,
            search_filter=s_filters,
            group_ids=[MOCK_GROUP_ID],
            limit=mock_limit
        )
        assert len(results)==1 and results[0].uuid == MOCK_UUID_COMMUNITY_1

        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        actual_params = mock_kuzu_connection.execute.call_args[1]

        assert "MATCH (c:Community)" in query_str
        assert "CONTAINS(lower(c.summary), $query_str_lower)" in query_str
        assert f"c.created_at {s_filters.created_at_filter.operator.value} $c_created_at_date" in query_str
        assert actual_params['c_created_at_date'] == MOCK_FILTER_DATETIME

    async def test_community_similarity_search(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_vector=[.1,.2,.3]; mock_limit=1; min_score=0.7
        s_filters = SearchFilters(group_id_filter="specific-community-group") # Test group_id from SearchFilter

        mock_ps=MagicMock();mock_kuzu_connection.prepare.return_value=mock_ps;mock_qr=MagicMock()
        cols = ["c.uuid", "c.name", "c.group_id", "c.created_at", "c.summary", "c.name_embedding", "score"]
        # Adjust mock data for the filter
        community_dict_match = {
            **MOCK_KUZU_COMMUNITY_NODE_DICT,
            "group_id": "specific-community-group", # Matches s_filters.group_id_filter
            "name_embedding": search_vector # Ensure it has the embedding
        }
        row1 = [community_dict_match[k.split('.')[1] if k != "score" else "uuid"] for k in cols]; row1[-1]=0.8
        configure_mock_query_result(mock_qr,cols,[row1]);mock_kuzu_connection.execute.return_value=mock_qr

        results = await kuzudb_provider.community_similarity_search(
            search_vector=search_vector,
            search_filter=s_filters,
            group_ids=[MOCK_GROUP_ID], # This will also be ANDed if present
            limit=mock_limit,
            min_score=min_score
        )
        assert len(results)==1

        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        actual_params = mock_kuzu_connection.execute.call_args[1]

        assert "cosine_similarity(c.name_embedding, $s_vec) >= $min_s" in query_str
        assert "c.group_id = $c_group_id_filter" in query_str # From SearchFilters
        assert actual_params['c_group_id_filter'] == "specific-community-group"
        # If group_ids method parameter is also used, it would add "c.group_id IN $gids"
        assert "c.group_id IN $gids" in query_str
        assert actual_params['gids'] == [MOCK_GROUP_ID]


    # --- Deletion Method Tests ---
    async def test_delete_node(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        node_uuid_to_delete = "delete-me-node-uuid"
        mock_ps = MagicMock()
        mock_kuzu_connection.prepare.return_value = mock_ps
        mock_qr_empty = MagicMock()
        configure_mock_query_result(mock_qr_empty, [], [], has_next=False) # Simulate no results or simple ack
        mock_kuzu_connection.execute.return_value = mock_qr_empty

        await kuzudb_provider.delete_node(node_uuid_to_delete)

        # Verify calls based on the implementation detail of trying to delete from all tables
        # and deleting relationships first.
        expected_deletion_attempts = 0
        node_tables = ["Entity", "Episodic", "Community"]
        rel_tables = ["MENTIONS", "RELATES_TO", "HAS_MEMBER"]

        calls_args_list = mock_kuzu_connection.prepare.call_args_list

        # Check if relationship deletion queries were prepared
        for node_table in node_tables:
            for rel_table in rel_tables:
                # MATCH (n:table {uuid: $uuid})-[r:rel_table]-() DELETE r
                assert any(f"MATCH (n:{node_table} {{uuid: $uuid}})-[r:{rel_table}]-() DELETE r" in call[0][0] for call in calls_args_list)
                # MATCH (n:table {uuid: $uuid})<-[r:rel_table]-() DELETE r
                assert any(f"MATCH (n:{node_table} {{uuid: $uuid}})<-[r:{rel_table}]-() DELETE r" in call[0][0] for call in calls_args_list)
            # Check node deletion query
            assert any(f"MATCH (n:{node_table} {{uuid: $uuid}}) DELETE n" in call[0][0] for call in calls_args_list)
            expected_deletion_attempts += (len(rel_tables) * 2) + 1

        assert mock_kuzu_connection.prepare.call_count == expected_deletion_attempts
        # All execute calls should use the uuid
        for call_kwargs in mock_kuzu_connection.execute.call_args_list:
            assert call_kwargs[1]['uuid'] == node_uuid_to_delete


    async def test_delete_nodes_by_group_id(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        group_id_to_delete = "delete-this-group"
        node_type_to_delete = "Entity" # Test for Entity, similar for others

        mock_ps = MagicMock()
        mock_kuzu_connection.prepare.return_value = mock_ps
        mock_qr_empty = MagicMock()
        configure_mock_query_result(mock_qr_empty, [], [], has_next=False)
        mock_kuzu_connection.execute.return_value = mock_qr_empty

        await kuzudb_provider.delete_nodes_by_group_id(group_id_to_delete, node_type_to_delete)

        calls_args_list = mock_kuzu_connection.prepare.call_args_list
        rel_tables = ["MENTIONS", "RELATES_TO", "HAS_MEMBER"]
        expected_prepare_calls = (len(rel_tables) * 2) + 1 # rel deletions + node deletion

        # Check relationship deletion queries
        for rel_table in rel_tables:
            assert any(f"MATCH (n:{node_type_to_delete} {{group_id: $gid}})-[r:{rel_table}]-() DELETE r" in call[0][0] for call in calls_args_list)
            assert any(f"MATCH (n:{node_type_to_delete} {{group_id: $gid}})<-[r:{rel_table}]-() DELETE r" in call[0][0] for call in calls_args_list)
        # Check node deletion query
        assert any(f"MATCH (n:{node_type_to_delete} {{group_id: $gid}}) DELETE n" in call[0][0] for call in calls_args_list)

        assert mock_kuzu_connection.prepare.call_count == expected_prepare_calls
        for call_kwargs in mock_kuzu_connection.execute.call_args_list:
            assert call_kwargs[1]['gid'] == group_id_to_delete

    async def test_delete_edge(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        edge_uuid_to_delete = "delete-me-edge-uuid"
        mock_ps = MagicMock()
        mock_kuzu_connection.prepare.return_value = mock_ps
        mock_qr_empty = MagicMock()
        configure_mock_query_result(mock_qr_empty, [], [], has_next=False)
        mock_kuzu_connection.execute.return_value = mock_qr_empty

        await kuzudb_provider.delete_edge(edge_uuid_to_delete)

        calls_args_list = mock_kuzu_connection.prepare.call_args_list
        rel_tables = ["MENTIONS", "RELATES_TO", "HAS_MEMBER"]
        expected_prepare_calls = len(rel_tables) # Tries each table type

        for rel_table in rel_tables:
             assert any(f"MATCH ()-[r:{rel_table} {{uuid: $uuid}}]-() DELETE r" in call[0][0] for call in calls_args_list)

        assert mock_kuzu_connection.prepare.call_count == expected_prepare_calls
        for call_kwargs in mock_kuzu_connection.execute.call_args_list:
            assert call_kwargs[1]['uuid'] == edge_uuid_to_delete

    # --- Schema and Data Management Tests ---
    async def test_build_indices_and_constraints_delete_false(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps = MagicMock()
        mock_kuzu_connection.prepare.return_value = mock_ps # All prepare calls return this
        mock_qr = MagicMock() # All execute calls return this
        configure_mock_query_result(mock_qr, [], [], has_next=False)
        mock_kuzu_connection.execute.return_value = mock_qr

        await kuzudb_provider.build_indices_and_constraints(delete_existing=False)

        prepared_queries = [call[0][0] for call in mock_kuzu_connection.prepare.call_args_list]

        # Check for CREATE TABLE queries (exact string match might be too brittle, check for key parts)
        assert any("CREATE NODE TABLE Entity" in q for q in prepared_queries)
        assert any("CREATE NODE TABLE Episodic" in q for q in prepared_queries)
        assert any("CREATE NODE TABLE Community" in q for q in prepared_queries)
        assert any("CREATE REL TABLE MENTIONS" in q for q in prepared_queries)
        assert any("CREATE REL TABLE RELATES_TO" in q for q in prepared_queries)
        assert any("CREATE REL TABLE HAS_MEMBER" in q for q in prepared_queries)

        # Ensure DROP TABLE was not called
        assert not any("DROP TABLE" in q for q in prepared_queries)

        # Kuzu schema creation is usually done by execute not prepare
        # The provider calls self.get_session().execute()
        # Let's verify the execute calls on the raw connection object
        # For this provider, `execute_query` calls `prepare` then `execute` on connection
        # The `build_indices_and_constraints` calls `conn.execute()` directly.
        # So, we need to inspect `mock_kuzu_connection.execute` (the raw one, not the one from execute_query)

        # Resetting prepare mock as it's not used by conn.execute()
        mock_kuzu_connection.prepare.reset_mock()

        # The fixture's mock_kuzu_connection.execute is the one we want to check
        # if build_indices_and_constraints uses the raw connection.execute.
        # The provider's execute_query uses prepare().execute().
        # build_indices_and_constraints uses get_session().execute() which is the raw execute.

        executed_direct_queries = [call[0][0] for call in mock_kuzu_connection.execute.call_args_list if isinstance(call[0][0], str)]

        assert any("CREATE NODE TABLE Entity" in q for q in executed_direct_queries)
        assert any("CREATE NODE TABLE Episodic" in q for q in executed_direct_queries)
        assert any("CREATE NODE TABLE Community" in q for q in executed_direct_queries)
        assert any("CREATE REL TABLE MENTIONS" in q for q in executed_direct_queries)
        assert any("CREATE REL TABLE RELATES_TO" in q for q in executed_direct_queries)
        assert any("CREATE REL TABLE HAS_MEMBER" in q for q in executed_direct_queries)
        assert not any("DROP TABLE" in q for q in executed_direct_queries)
        # Total 6 create table statements
        assert len(executed_direct_queries) == 6


    async def test_build_indices_and_constraints_delete_true(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        # Reset mocks from previous tests if any state is shared via fixture scope
        mock_kuzu_connection.reset_mock() # Resets prepare, execute etc.

        mock_ps = MagicMock()
        mock_kuzu_connection.prepare.return_value = mock_ps
        mock_qr = MagicMock()
        configure_mock_query_result(mock_qr, [], [], has_next=False)
        mock_kuzu_connection.execute.return_value = mock_qr # For direct execute calls

        await kuzudb_provider.build_indices_and_constraints(delete_existing=True)

        executed_direct_queries = [call[0][0] for call in mock_kuzu_connection.execute.call_args_list if isinstance(call[0][0], str)]

        table_names = ["Entity", "Episodic", "Community", "MENTIONS", "RELATES_TO", "HAS_MEMBER"]
        for name in table_names:
            assert any(f"DROP TABLE {name}" in q for q in executed_direct_queries)
            if "NODE" in q or "REL" in q: # Check create only for actual table creation lines
                 assert any(f"CREATE NODE TABLE {name}" in q or f"CREATE REL TABLE {name}" in q for q in executed_direct_queries)

        # Total 6 drop + 6 create = 12 direct execute calls
        assert len(executed_direct_queries) == 12

    async def test_clear_data_all(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_kuzu_connection.reset_mock()
        mock_ps = MagicMock()
        mock_kuzu_connection.prepare.return_value = mock_ps
        mock_qr = MagicMock()
        configure_mock_query_result(mock_qr, [], [], has_next=False)
        mock_kuzu_connection.execute.return_value = mock_qr

        # Patch build_indices_and_constraints to verify it's called
        with patch.object(kuzudb_provider, 'build_indices_and_constraints', new_callable=AsyncMock) as mock_build_indices:
            await kuzudb_provider.clear_data(group_ids=None)
            mock_build_indices.assert_called_once_with(delete_existing=True)

    async def test_clear_data_with_group_ids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_kuzu_connection.reset_mock()
        group_ids_to_clear = ["group1", "group2"]

        mock_ps = MagicMock()
        mock_kuzu_connection.prepare.return_value = mock_ps
        mock_qr = MagicMock()
        configure_mock_query_result(mock_qr, [], [], has_next=False)
        # For execute_query, which uses prepare().execute()
        mock_kuzu_connection.execute.return_value = mock_qr # This is for the prepared statement's execute

        await kuzudb_provider.clear_data(group_ids=group_ids_to_clear)

        prepared_queries = [call[0][0] for call in mock_kuzu_connection.prepare.call_args_list]
        executed_params_list = [call[1] for call in mock_kuzu_connection.execute.call_args_list]

        node_tables = ["Entity", "Episodic", "Community"]
        rel_tables = ["MENTIONS", "RELATES_TO", "HAS_MEMBER"]

        expected_prepare_calls = 0
        # For each group_id
        for gid in group_ids_to_clear:
            # 1. Delete rels WITH the group_id
            for rel_table in rel_tables:
                assert any(f"MATCH ()-[r:{rel_table} {{group_id: $gid}}]-() DELETE r" in q for q in prepared_queries)
                assert any(params['gid'] == gid for params in executed_params_list if 'gid' in params)
                expected_prepare_calls +=1

            # 2. Delete rels connected TO nodes with the group_id
            for node_table in node_tables:
                for rel_table in rel_tables:
                    assert any(f"MATCH (n:{node_table} {{group_id: $gid}})-[r:{rel_table}]-() DELETE r" in q for q in prepared_queries)
                    assert any(params['gid'] == gid for params in executed_params_list if 'gid' in params)
                    expected_prepare_calls +=1
                    assert any(f"MATCH ()-[r:{rel_table}]-(n:{node_table} {{group_id: $gid}}) DELETE r" in q for q in prepared_queries)
                    assert any(params['gid'] == gid for params in executed_params_list if 'gid' in params)
                    expected_prepare_calls +=1

            # 3. Delete nodes
            for node_table in node_tables:
                assert any(f"MATCH (n:{node_table} {{group_id: $gid}}) DELETE n" in q for q in prepared_queries)
                assert any(params['gid'] == gid for params in executed_params_list if 'gid' in params)
                expected_prepare_calls +=1

        assert mock_kuzu_connection.prepare.call_count == expected_prepare_calls
        assert mock_kuzu_connection.execute.call_count == expected_prepare_calls

    # --- Search Method Tests (Full-text) ---
    async def test_node_fulltext_search_with_search_filters_and_group_ids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_query = "Test Entity"
        mock_limit = 3
        specific_group_ids = [MOCK_GROUP_ID] # Group IDs passed directly to method

        # SearchFilters object with its own filters
        s_filters = SearchFilters(
            created_at_filter=DateFilter(date=MOCK_FILTER_DATETIME, operator=ComparisonOperator.LESS_THAN),
            group_id_filter="filter-group-id" # This tests group_id from SearchFilters object
        )

        mock_ps = MagicMock(name="ps_node_fts_sf_gid"); mock_kuzu_connection.prepare.return_value = mock_ps
        mock_qr = MagicMock(name="qr_node_fts_sf_gid")
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes"]
        # Simulate a node that matches all criteria
        node_dict_match = {
            **MOCK_KUZU_ENTITY_NODE_DICT,
            "group_id": MOCK_GROUP_ID, # Matches specific_group_ids
            # created_at needs to be before MOCK_FILTER_DATETIME for LESS_THAN
        }
        # To make it satisfy the group_id_filter from SearchFilters as well, we'd need an OR in query or adjust mock data.
        # The current _apply_search_filters_to_query_basic ANDs all conditions.
        # So, for this test, we assume the node's group_id also matches s_filters.group_id_filter
        # Let's adjust the mock data to reflect this for a successful hit.
        node_dict_match["group_id"] = "filter-group-id" # To match s_filters.group_id_filter
        # And ensure specific_group_ids also contains this for the n.group_id IN $gids part
        specific_group_ids_for_test = ["filter-group-id"]


        row1 = [node_dict_match[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [row1]); mock_kuzu_connection.execute.return_value = mock_qr

        results = await kuzudb_provider.node_fulltext_search(
            query=search_query,
            search_filter=s_filters,
            group_ids=specific_group_ids_for_test, # Method's own group_ids
            limit=mock_limit
        )
        assert len(results) == 1

        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        actual_params = mock_kuzu_connection.execute.call_args[1]

        # Check for FTS part
        assert f"CONTAINS(lower(n.name), $query_str_lower)" in query_str
        assert actual_params['query_str_lower'] == search_query.lower()

        # Check for group_ids from method parameter (takes precedence or is ANDed)
        assert "n.group_id IN $gids" in query_str
        assert actual_params['gids'] == specific_group_ids_for_test

        # Check for created_at filter from SearchFilters
        assert f"n.created_at {s_filters.created_at_filter.operator.value} $n_created_at_date" in query_str
        assert actual_params['n_created_at_date'] == MOCK_FILTER_DATETIME

        # Check for group_id filter from SearchFilters
        assert "n.group_id = $n_group_id_filter" in query_str
        assert actual_params['n_group_id_filter'] == "filter-group-id"

        assert "LIMIT $limit_val" in query_str
        assert actual_params['limit_val'] == mock_limit

    async def test_edge_fulltext_search_with_search_filters(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_query = "Test Fact"
        s_filters = SearchFilters(
            created_at_filter=DateFilter(date=MOCK_FILTER_DATETIME, operator=ComparisonOperator.EQUALS)
        )
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        cols = ["r.uuid", "r.name", "r.group_id", "r.fact", "r.fact_embedding", "r.episodes", "r.created_at", "r.expired_at", "r.valid_at", "r.invalid_at", "r.attributes", "source_node_uuid", "target_node_uuid"]
        row1 = [MOCK_KUZU_ENTITY_EDGE_DICT[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr, cols, [row1]); mock_kuzu_connection.execute.return_value = mock_qr

        results = await kuzudb_provider.edge_fulltext_search(query=search_query, search_filter=s_filters, limit=1)
        assert len(results) == 1
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        actual_params = mock_kuzu_connection.execute.call_args[1]
        assert f"r.created_at {s_filters.created_at_filter.operator.value} $r_created_at_date" in query_str
        assert actual_params['r_created_at_date'] == MOCK_FILTER_DATETIME

    async def test_episode_fulltext_search_with_search_filters(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_query = "Episode Content"
        s_filters = SearchFilters(
            valid_at_filter=DateFilter(date=MOCK_FILTER_DATETIME, operator=ComparisonOperator.GREATER_THAN)
        )
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        cols = ["e.uuid", "e.name", "e.group_id", "e.source", "e.source_description", "e.content", "e.valid_at", "e.created_at", "e.entity_edges"]
        row1 = [MOCK_KUZU_EPISODIC_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [row1]); mock_kuzu_connection.execute.return_value = mock_qr

        results = await kuzudb_provider.episode_fulltext_search(query=search_query, search_filter=s_filters, limit=1)
        assert len(results) == 1
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        actual_params = mock_kuzu_connection.execute.call_args[1]
        assert f"e.valid_at {s_filters.valid_at_filter.operator.value} $e_valid_at_date" in query_str
        assert actual_params['e_valid_at_date'] == MOCK_FILTER_DATETIME

    # Community fulltext search does not currently accept SearchFilters in its signature in KuzuDBProvider
    # So, no specific SearchFilter test for it beyond what's already there.

    # --- Search Method Tests (Similarity) ---
    async def test_node_similarity_search_with_search_filters(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_vector = [0.1,0.2,0.3]
        s_filters = SearchFilters(
            created_at_filter=DateFilter(date=MOCK_FILTER_DATETIME, operator=ComparisonOperator.LESS_THAN_OR_EQUALS)
        )
        mock_ps=MagicMock();mock_kuzu_connection.prepare.return_value=mock_ps;mock_qr=MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes", "score"]
        row1 = [MOCK_KUZU_ENTITY_NODE_DICT[k.split('.')[1] if k != "score" else "uuid"] for k in cols]; row1[-1]=0.8
        configure_mock_query_result(mock_qr,cols,[row1]);mock_kuzu_connection.execute.return_value=mock_qr

        results = await kuzudb_provider.node_similarity_search(search_vector=search_vector, search_filter=s_filters, limit=1)
        assert len(results)==1
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        actual_params = mock_kuzu_connection.execute.call_args[1]
        assert f"n.created_at {s_filters.created_at_filter.operator.value} $n_created_at_date" in query_str
        assert actual_params['n_created_at_date'] == MOCK_FILTER_DATETIME
        assert "cosine_similarity(n.name_embedding, $s_vec)" in query_str

    async def test_edge_similarity_search_with_search_filters(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_vector=[.1,.2,.3]
        s_filters = SearchFilters(
            group_id_filter="filter-group-for-edge-sim"
        )
        mock_ps=MagicMock();mock_kuzu_connection.prepare.return_value=mock_ps;mock_qr=MagicMock()
        cols = ["r.uuid", "r.name", "r.group_id", "r.fact", "r.fact_embedding", "r.episodes", "r.created_at", "r.expired_at", "r.valid_at", "r.invalid_at", "r.attributes", "source_node_uuid", "target_node_uuid", "score"]
        # Adjust mock data to match the group_id_filter from SearchFilters
        edge_dict_match = {**MOCK_KUZU_ENTITY_EDGE_DICT, "group_id": "filter-group-for-edge-sim"}
        row1 = [edge_dict_match[k.split('.')[1] if k != "score" and '.' in k else k] for k in cols]; row1[-1]=0.9
        configure_mock_query_result(mock_qr,cols,[row1]);mock_kuzu_connection.execute.return_value=mock_qr

        results = await kuzudb_provider.edge_similarity_search(search_vector=search_vector, search_filter=s_filters, limit=1)
        assert len(results)==1
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        actual_params = mock_kuzu_connection.execute.call_args[1]
        assert "r.group_id = $r_group_id_filter" in query_str
        assert actual_params['r_group_id_filter'] == "filter-group-for-edge-sim"
        assert "cosine_similarity(r.fact_embedding, $s_vec)" in query_str

    # Community similarity search does not currently accept SearchFilters in its signature
    # So, no specific SearchFilter test for it beyond what's already there.

    # --- BFS Search Tests ---
    async def test_node_bfs_search_basic(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        origin_uuids = [MOCK_UUID_ENTITY_1]
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        # BFS result cols for peer node
        cols = ["peer.uuid", "peer.name", "peer.group_id", "peer.created_at", "peer.summary", "peer.name_embedding", "peer.attributes"]
        # Simulate peer node being MOCK_KUZU_ENTITY_NODE_DICT_2
        row1 = [MOCK_KUZU_ENTITY_NODE_DICT_2[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [row1]); mock_kuzu_connection.execute.return_value = mock_qr

        results = await kuzudb_provider.node_bfs_search(
            bfs_origin_node_uuids=origin_uuids,
            search_filter=EMPTY_SEARCH_FILTERS,
            bfs_max_depth=2,
            limit=1
        )
        assert len(results) == 1
        assert results[0].uuid == MOCK_UUID_ENTITY_2
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert f"MATCH (origin:Entity)-[rels*1..2 BFS]->(peer:Entity)" in query_str # Max depth 2
        assert "origin.uuid IN $origin_uuids" in query_str
        assert "origin.uuid <> peer.uuid" in query_str # Default filter
        assert mock_kuzu_connection.execute.call_args[1]['origin_uuids'] == origin_uuids

    async def test_node_bfs_search_with_edge_types_filter(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        origin_uuids = [MOCK_UUID_ENTITY_1]
        s_filters = SearchFilters(edge_types=["RELATES_TO_A", "RELATES_TO_B"])
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        configure_mock_query_result(mock_qr, [], [], has_next=False); mock_kuzu_connection.execute.return_value = mock_qr # No results needed for query check

        await kuzudb_provider.node_bfs_search(
            bfs_origin_node_uuids=origin_uuids,
            search_filter=s_filters,
            bfs_max_depth=1,
            limit=1
        )
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "MATCH (origin:Entity)-[rels:RELATES_TO_A|RELATES_TO_B*1..1 BFS]->(peer:Entity)" in query_str

    async def test_edge_bfs_search_basic(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        origin_uuids = [MOCK_UUID_ENTITY_1]
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        # BFS result cols for edge r_other
        cols = ["r.uuid", "r.name", "r.group_id", "r.fact", "r.fact_embedding", "r.episodes", "r.created_at", "r.expired_at", "r.valid_at", "r.invalid_at", "r.attributes", "source_node_uuid", "target_node_uuid"]
        # Simulate edge being MOCK_KUZU_ENTITY_EDGE_DICT
        row1 = [MOCK_KUZU_ENTITY_EDGE_DICT[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr, cols, [row1]); mock_kuzu_connection.execute.return_value = mock_qr

        results = await kuzudb_provider.edge_bfs_search(
            bfs_origin_node_uuids=origin_uuids,
            bfs_max_depth=1,
            search_filter=EMPTY_SEARCH_FILTERS,
            limit=1
        )
        assert len(results) == 1
        assert results[0].uuid == MOCK_UUID_EDGE_1
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "MATCH (origin:Entity)-[rels*1..1 BFS]->(peer:Entity)" in query_str
        assert "UNWIND rels AS r" in query_str
        assert "RETURN DISTINCT r.uuid" in query_str
        assert mock_kuzu_connection.execute.call_args[1]['origin_uuids'] == origin_uuids

    async def test_edge_bfs_search_with_r_filter(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        origin_uuids = [MOCK_UUID_ENTITY_1]
        s_filters = SearchFilters(group_id_filter="bfs-edge-group") # Filter for the edge 'r'
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        configure_mock_query_result(mock_qr, [], [], has_next=False); mock_kuzu_connection.execute.return_value = mock_qr

        await kuzudb_provider.edge_bfs_search(
            bfs_origin_node_uuids=origin_uuids,
            bfs_max_depth=1,
            search_filter=s_filters,
            limit=1
        )
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        actual_params = mock_kuzu_connection.execute.call_args[1]
        # Check that the filter for 'r' is applied after UNWIND
        assert "UNWIND rels AS r" in query_str
        assert "WHERE r.group_id = $r_group_id_filter" in query_str # Filter on 'r'
        assert actual_params['r_group_id_filter'] == "bfs-edge-group"

    # --- Utility/Helper Method Tests ---
    async def test_retrieve_episodes(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        ref_time = MOCK_DATETIME_NATIVE
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        cols = ["e.uuid", "e.name", "e.group_id", "e.source", "e.source_description", "e.content", "e.valid_at", "e.created_at", "e.entity_edges"]
        row1 = [MOCK_KUZU_EPISODIC_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [row1]); mock_kuzu_connection.execute.return_value = mock_qr

        results = await kuzudb_provider.retrieve_episodes(
            reference_time=ref_time,
            last_n=1,
            group_ids=[MOCK_GROUP_ID],
            source=EpisodeType.text
        )
        assert len(results) == 1
        assert results[0].uuid == MOCK_UUID_EPISODIC_1
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        actual_params = mock_kuzu_connection.execute.call_args[1]
        assert "e.valid_at <= $ref_time" in query_str
        assert "e.group_id IN $gids" in query_str
        assert "e.source = $src" in query_str
        assert "ORDER BY e.valid_at DESC LIMIT $limit_n" in query_str
        assert actual_params['ref_time'] == ref_time
        assert actual_params['gids'] == [MOCK_GROUP_ID]
        assert actual_params['src'] == EpisodeType.text.value

    async def test_count_node_mentions(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        node_uuid = MOCK_UUID_ENTITY_1
        expected_count = 7
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        configure_mock_query_result(mock_qr, ["mention_count"], [[expected_count]]); mock_kuzu_connection.execute.return_value = mock_qr

        count = await kuzudb_provider.count_node_mentions(node_uuid)
        assert count == expected_count
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $node_uuid})" in query_str
        assert "RETURN count(DISTINCT e.uuid) AS mention_count" in query_str
        assert mock_kuzu_connection.execute.call_args[1]['node_uuid'] == node_uuid

    async def test_get_embeddings_for_nodes(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        nodes_in = [
            EntityNode(uuid=MOCK_UUID_ENTITY_1, name="N1"),
            EntityNode(uuid=MOCK_UUID_ENTITY_2, name="N2")
        ]
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        cols = ["uuid", "embedding"]
        rows = [
            [MOCK_UUID_ENTITY_1, MOCK_KUZU_ENTITY_NODE_DICT["name_embedding"]],
            [MOCK_UUID_ENTITY_2, None] # One node without embedding
        ]
        configure_mock_query_result(mock_qr, cols, rows); mock_kuzu_connection.execute.return_value = mock_qr

        results = await kuzudb_provider.get_embeddings_for_nodes(nodes_in)
        assert len(results) == 1 # Only one has embedding
        assert results[MOCK_UUID_ENTITY_1] == MOCK_KUZU_ENTITY_NODE_DICT["name_embedding"]
        assert MOCK_UUID_ENTITY_2 not in results
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "MATCH (n:Entity) WHERE n.uuid IN $uuids RETURN n.uuid AS uuid, n.name_embedding AS embedding" in query_str
        assert mock_kuzu_connection.execute.call_args[1]['uuids'] == [MOCK_UUID_ENTITY_1, MOCK_UUID_ENTITY_2]

    # Similar tests for get_embeddings_for_communities, get_embeddings_for_edges would follow

    async def test_get_mentioned_nodes(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        episodes_in = [EpisodicNode(uuid=MOCK_UUID_EPISODIC_1, name="Ep1")]
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes"]
        row1 = [MOCK_KUZU_ENTITY_NODE_DICT[k.split('.')[1]] for k in cols] # Assume episode mentions Entity1
        configure_mock_query_result(mock_qr, cols, [row1]); mock_kuzu_connection.execute.return_value = mock_qr

        results = await kuzudb_provider.get_mentioned_nodes(episodes_in)
        assert len(results) == 1
        assert results[0].uuid == MOCK_UUID_ENTITY_1
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "MATCH (e:Episodic)-[:MENTIONS]->(n:Entity) WHERE e.uuid IN $uuids RETURN DISTINCT" in query_str
        assert mock_kuzu_connection.execute.call_args[1]['uuids'] == [MOCK_UUID_EPISODIC_1]

    async def test_get_communities_by_nodes(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        nodes_in = [EntityNode(uuid=MOCK_UUID_ENTITY_1, name="N1")]
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        cols = ["c.uuid", "c.name", "c.group_id", "c.created_at", "c.summary", "c.name_embedding"]
        row1 = [MOCK_KUZU_COMMUNITY_NODE_DICT[k.split('.')[1]] for k in cols] # Assume Entity1 is member of Community1
        configure_mock_query_result(mock_qr, cols, [row1]); mock_kuzu_connection.execute.return_value = mock_qr

        results = await kuzudb_provider.get_communities_by_nodes(nodes_in)
        assert len(results) == 1
        assert results[0].uuid == MOCK_UUID_COMMUNITY_1
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity) WHERE n.uuid IN $uuids RETURN DISTINCT" in query_str
        assert mock_kuzu_connection.execute.call_args[1]['uuids'] == [MOCK_UUID_ENTITY_1]

    # --- Bulk Operations Test (Conceptual) ---
    @patch('tempfile.mkstemp', return_value=(0, 'dummy.parquet')) # Mock file creation
    @patch('os.close')
    @patch('os.remove')
    @patch('pandas.DataFrame.to_parquet') # Mock parquet writing
    async def test_add_nodes_and_edges_bulk(self, mock_to_parquet, mock_os_remove, mock_os_close, mock_mkstemp, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_embedder = AsyncMock(spec=EmbedderClient)
        # Simulate embedding generation for one node/edge that needs it
        mock_embedder.create = AsyncMock(return_value=[[0.01, 0.02, 0.03]])

        entity_nodes = [
            EntityNode(uuid="bulk-en1", name="Bulk Entity 1", group_id="g1"), # Will need embedding
            EntityNode(uuid="bulk-en2", name="Bulk Entity 2", group_id="g1", name_embedding=[.1,.2,.3]) # Has embedding
        ]
        entity_edges = [
            EntityEdge(uuid="bulk-ee1", name="R", source_node_uuid="bulk-en1", target_node_uuid="bulk-en2", fact="Bulk Fact 1") # Will need embedding
        ]
        episodic_nodes = [EpisodicNode(uuid="bulk-ep1", name="Bulk Episode 1", source=EpisodeType.text, group_id="g1")]
        episodic_edges = [EpisodicEdge(uuid="bulk-epe1", source_node_uuid="bulk-ep1", target_node_uuid="bulk-en1", group_id="g1")]

        mock_ps_copy = MagicMock(name="ps_copy_from")
        mock_qr_copy = MagicMock(name="qr_copy_from")
        configure_mock_query_result(mock_qr_copy,[],[],has_next=False)

        # Side effect for prepare: return ps_copy_from if query contains "COPY"
        def prepare_side_effect_for_copy(query_str):
            if "COPY" in query_str: return mock_ps_copy
            return MagicMock() # Default for other prepares if any
        mock_kuzu_connection.prepare.side_effect = prepare_side_effect_for_copy
        mock_kuzu_connection.execute.return_value = mock_qr_copy # For COPY execution

        await kuzudb_provider.add_nodes_and_edges_bulk(
            episodic_nodes=episodic_nodes, episodic_edges=episodic_edges,
            entity_nodes=entity_nodes, entity_edges=entity_edges,
            embedder=mock_embedder
        )

        # Verify embedder calls (1 for entity node, 1 for entity edge)
        assert mock_embedder.create.call_count == 2

        # Verify to_parquet calls (one for each type of data: Entity, Episodic, MENTIONS, RELATES_TO)
        assert mock_to_parquet.call_count == 4

        # Verify COPY queries were prepared and executed
        copy_prepare_calls = [call for call in mock_kuzu_connection.prepare.call_args_list if "COPY" in call[0][0]]
        assert len(copy_prepare_calls) == 4 # Entity, Episodic, MENTIONS, RELATES_TO

        copy_execute_calls = [call for call in mock_kuzu_connection.execute.call_args_list if call[0][0] == mock_ps_copy]
        assert len(copy_execute_calls) == 4

        # Example: Check one COPY query for correct options (e.g., Entity with ON_CONFLICT)
        entity_copy_query = next(q[0][0] for q in copy_prepare_calls if "COPY Entity FROM" in q[0][0])
        assert "(ON_CONFLICT (uuid) DO UPDATE SET" in entity_copy_query

        mentions_copy_query = next(q[0][0] for q in copy_prepare_calls if "COPY MENTIONS FROM" in q[0][0])
        assert "(ON_CONFLICT (uuid) DO UPDATE SET" not in mentions_copy_query # Rels shouldn't have it typically

    # Add more tests for RRF, get_relevant_*, get_edge_invalidation_candidates, etc.
    # These will often involve mocking the underlying search methods of the provider itself.
    # Example for get_relevant_nodes_rrf:
    async def test_get_relevant_nodes_rrf(self, kuzudb_provider: KuzuDBProvider):
        input_node = EntityNode(uuid="center-uuid", name="Center", group_id="g1", name_embedding=[.1,.1,.1])

        # Mock node_similarity_search
        mock_sim_node1 = EntityNode(uuid="sim-node1", name="Similar1", group_id="g1")
        kuzudb_provider.node_similarity_search = AsyncMock(return_value=[mock_sim_node1])

        # Mock node_fulltext_search
        mock_fts_node1 = EntityNode(uuid="fts-node1", name="FTS1", group_id="g1")
        mock_fts_node2 = EntityNode(uuid="sim-node1", name="Similar1_fts", group_id="g1") # Also found by FTS
        kuzudb_provider.node_fulltext_search = AsyncMock(return_value=[mock_fts_node1, mock_fts_node2])

        # Mock get_entity_nodes_by_uuids for the final fetch
        # RRF order might be: sim-node1 (from both), fts-node1
        # Let's assume RRF results in uuids: ["sim-node1", "fts-node1"]
        async def mock_get_by_uuids(uuids: List[str]):
            res = []
            if "sim-node1" in uuids: res.append(mock_sim_node1) # Use the original object for simplicity
            if "fts-node1" in uuids: res.append(mock_fts_node1)
            # Ensure order matches input uuids for this mock
            ordered_res = []
            for u in uuids:
                if u == "sim-node1": ordered_res.append(mock_sim_node1)
                elif u == "fts-node1": ordered_res.append(mock_fts_node1)
            return ordered_res
        kuzudb_provider.get_entity_nodes_by_uuids = AsyncMock(side_effect=mock_get_by_uuids)

        results_list = await kuzudb_provider.get_relevant_nodes_rrf(
            nodes=[input_node], search_filter=EMPTY_SEARCH_FILTERS, limit=2
        )

        assert len(results_list) == 1
        final_nodes = results_list[0]
        assert len(final_nodes) <= 2 # Limit

        # Check that sub-methods were called
        kuzudb_provider.node_similarity_search.assert_called_once()
        kuzudb_provider.node_fulltext_search.assert_called_once()
        kuzudb_provider.get_entity_nodes_by_uuids.assert_called_once()

        # Verify RRF logic (simplified check: sim-node1 should be ranked high)
        if final_nodes:
            assert "sim-node1" in [n.uuid for n in final_nodes]
            # A more detailed test would calculate exact RRF scores and check order.
            # For now, ensuring the components are called and results are combined is a good start.
            # Example check for order if limit allows both:
            if len(final_nodes) == 2:
                 # sim-node1 appears in vector (rank 0) and fts (rank 1)
                 # fts-node1 appears in fts (rank 0)
                 # RRF for sim-node1: 1/(60+0+1) + 1/(60+1+1) approx 0.0163 + 0.0161 = 0.0324
                 # RRF for fts-node1: 1/(60+0+1) approx 0.0163
                 # So sim-node1 should be first
                assert final_nodes[0].uuid == "sim-node1"
                assert final_nodes[1].uuid == "fts-node1"

    async def test_node_search_with_attribute_filter(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_query = "NodeWithSpecificAttribute" # For FTS part
        # Attribute filter: looking for '{"special_key":"special_value"}' in attributes string
        s_filters = SearchFilters(
            attribute_filters=[
                AttributeFilter(key="special_key", value="special_value", operator=ComparisonOperator.CONTAINS)
            ]
        )

        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps
        mock_qr = MagicMock()

        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes"]
        # Mock node data that would match this attribute filter
        matching_node_attributes = json.dumps({"prop1": "val1", "special_key": "special_value", "prop2": "val2"})
        node_dict_match = {
            **MOCK_KUZU_ENTITY_NODE_DICT,
            "uuid": "attr-node-1",
            "name": search_query, # Matches FTS
            "attributes": matching_node_attributes # Matches attribute filter
        }
        row1 = [node_dict_match[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [row1]); mock_kuzu_connection.execute.return_value = mock_qr

        # Test with node_fulltext_search
        results_fts = await kuzudb_provider.node_fulltext_search(query=search_query, search_filter=s_filters, limit=1)
        assert len(results_fts) == 1
        assert results_fts[0].uuid == "attr-node-1"

        query_str_fts = mock_kuzu_connection.prepare.call_args[0][0]
        params_fts = mock_kuzu_connection.execute.call_args[1]
        assert "CONTAINS(lower(n.attributes), lower($n_attr_val_0))" in query_str_fts
        assert params_fts["n_attr_val_0"] == "special_value" # As per current _apply_search_filters_to_query_basic

        # Reset mocks for next search type
        mock_kuzu_connection.prepare.reset_mock()
        mock_kuzu_connection.execute.reset_mock()
        mock_ps_sim = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps_sim
        mock_qr_sim = MagicMock()
        # Similarity search also returns 'score'
        sim_cols = cols + ["score"]
        sim_row1 = row1 + [0.9] # Add a mock score
        configure_mock_query_result(mock_qr_sim, sim_cols, [sim_row1]); mock_kuzu_connection.execute.return_value = mock_qr_sim

        # Test with node_similarity_search
        search_vector = [0.1,0.1,0.1]
        node_dict_match["name_embedding"] = search_vector # Ensure it has an embedding for the search

        results_sim = await kuzudb_provider.node_similarity_search(search_vector=search_vector, search_filter=s_filters, limit=1, min_score=0.8)
        assert len(results_sim) == 1
        assert results_sim[0].uuid == "attr-node-1"

        query_str_sim = mock_kuzu_connection.prepare.call_args[0][0]
        params_sim = mock_kuzu_connection.execute.call_args[1]
        assert "CONTAINS(lower(n.attributes), lower($n_attr_val_0))" in query_str_sim
        assert params_sim["n_attr_val_0"] == "special_value"
        assert "cosine_similarity(n.name_embedding, $s_vec)" in query_str_sim

    async def test_get_relevant_edges_rrf(self, kuzudb_provider: KuzuDBProvider):
        input_edge = EntityEdge(
            uuid="center-edge-uuid", name="R", source_node_uuid="s1", target_node_uuid="t1",
            fact="Central Fact", group_id="g1", fact_embedding=[.2,.2,.2]
        )

        mock_sim_edge1 = EntityEdge(uuid="sim-edge1", name="R", source_node_uuid="s2", target_node_uuid="t2", fact="Similar Edge 1", group_id="g1")
        kuzudb_provider.edge_similarity_search = AsyncMock(return_value=[mock_sim_edge1])

        mock_fts_edge1 = EntityEdge(uuid="fts-edge1", name="R", source_node_uuid="s3", target_node_uuid="t4", fact="FTS Edge 1", group_id="g1")
        mock_fts_edge2 = EntityEdge(uuid="sim-edge1", name="R", source_node_uuid="s2", target_node_uuid="t2", fact="Similar Edge 1 FTS", group_id="g1")
        kuzudb_provider.edge_fulltext_search = AsyncMock(return_value=[mock_fts_edge1, mock_fts_edge2])

        async def mock_get_edges_by_uuids(uuids: List[str]):
            res = []
            if "sim-edge1" in uuids: res.append(mock_sim_edge1)
            if "fts-edge1" in uuids: res.append(mock_fts_edge1)
            ordered_res = []
            for u in uuids: # Ensure order for RRF check
                if u == "sim-edge1": ordered_res.append(mock_sim_edge1)
                elif u == "fts-edge1": ordered_res.append(mock_fts_edge1)
            return ordered_res
        kuzudb_provider.get_entity_edges_by_uuids = AsyncMock(side_effect=mock_get_edges_by_uuids)

        results_list = await kuzudb_provider.get_relevant_edges(
            edges=[input_edge], search_filter=EMPTY_SEARCH_FILTERS, limit=2, min_score=0.1 # min_score for similarity search
        )
        assert len(results_list) == 1
        final_edges = results_list[0]
        assert len(final_edges) <= 2

        kuzudb_provider.edge_similarity_search.assert_called_once()
        kuzudb_provider.edge_fulltext_search.assert_called_once()
        kuzudb_provider.get_entity_edges_by_uuids.assert_called_once()
        if len(final_edges) == 2: # Check order if both made it
            assert final_edges[0].uuid == "sim-edge1"
            assert final_edges[1].uuid == "fts-edge1"

    async def test_get_edge_invalidation_candidates(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        input_edge_ref = EntityEdge(
            uuid="ref-edge-uuid", name="REF_REL",
            source_node_uuid=MOCK_UUID_ENTITY_1, target_node_uuid=MOCK_UUID_ENTITY_2,
            fact="Reference Fact", fact_embedding=[0.5, 0.5, 0.5], group_id=MOCK_GROUP_ID
        )
        min_score_threshold = 0.7
        limit_val = 5

        # Mock Kuzu query result for this specific test
        mock_ps_invalid = MagicMock(name="ps_edge_invalid")
        mock_qr_invalid = MagicMock(name="qr_edge_invalid")

        # Candidate edge that should be returned by the query
        candidate_edge_data = {
            **MOCK_KUZU_ENTITY_EDGE_DICT, # Use base mock
            "uuid": "candidate-edge-uuid",
            "fact_embedding": [0.6, 0.6, 0.6], # Similar enough
            "source_node_uuid": MOCK_UUID_ENTITY_1, # Connected to ref source
            "target_node_uuid": "other-node-uuid",
            "score": 0.8 # Assume this score is calculated by Kuzu
        }
        # Ensure all keys for _kuzudb_to_entity_edge are present
        cols_for_edge = ["r_other.uuid", "r_other.name", "r_other.group_id", "r_other.fact", "r_other.fact_embedding", "r_other.episodes", "r_other.created_at", "r_other.expired_at", "r_other.valid_at", "r_other.invalid_at", "r_other.attributes", "n1.uuid AS source_node_uuid", "n2.uuid AS target_node_uuid", "score"]

        # Map candidate_edge_data to the column names Kuzu returns
        # Note: The query uses r_other, n1, n2 aliases. Parser expects specific keys.
        # The parser _kuzudb_to_entity_edge expects keys like 'uuid', 'name', 'source_node_uuid', etc.
        # The RETURN statement in the provider method does:
        # "r_other.uuid, ..., n1.uuid AS source_node_uuid, n2.uuid AS target_node_uuid, score"
        # So, the keys in the mocked dict should match these post-aliasing names.

        mock_row_values = [
            candidate_edge_data["uuid"], candidate_edge_data["name"], candidate_edge_data["group_id"],
            candidate_edge_data["fact"], candidate_edge_data["fact_embedding"], candidate_edge_data.get("episodes",[]),
            candidate_edge_data["created_at"], candidate_edge_data.get("expired_at"),
            candidate_edge_data.get("valid_at"), candidate_edge_data.get("invalid_at"),
            candidate_edge_data["attributes"], candidate_edge_data["source_node_uuid"], # n1.uuid (source of r_other)
            candidate_edge_data["target_node_uuid"], # n2.uuid (target of r_other)
            candidate_edge_data["score"]
        ]

        # These are the column names as returned by Kuzu in the query within get_edge_invalidation_candidates
        # Need to match the order and naming in the RETURN clause of that method's query
        # cols_returned_by_query = ["r_other.uuid", "r_other.name", ..., "source_node_uuid", "target_node_uuid", "score"]
        # For simplicity, assume the _kuzudb_to_entity_edge can handle the dict directly if keys match.
        # The execute_query mock path:
        # provider.execute_query -> connection.prepare -> connection.execute -> returns QueryResult
        # QueryResult.get_as_list() -> list of lists
        # QueryResult.get_column_names() -> list of strings
        # Then these are zipped into dicts. So the mock_row_values need to match cols_returned_by_query.

        # Let's use the actual column names from the method's query for clarity in mocking
        # The method defines:
        # cols = "r_other.uuid, r_other.name, ..., n1.uuid AS source_node_uuid, n2.uuid AS target_node_uuid, score"
        # These are the keys that will be in the dicts passed to _kuzudb_to_entity_edge
        mock_result_dict = {
            "uuid": candidate_edge_data["uuid"], "name": candidate_edge_data["name"], "group_id": candidate_edge_data["group_id"],
            "fact": candidate_edge_data["fact"], "fact_embedding": candidate_edge_data["fact_embedding"],
            "episodes": candidate_edge_data.get("episodes", []), "created_at": candidate_edge_data["created_at"],
            "expired_at": candidate_edge_data.get("expired_at"), "valid_at": candidate_edge_data.get("valid_at"),
            "invalid_at": candidate_edge_data.get("invalid_at"), "attributes": candidate_edge_data["attributes"],
            "source_node_uuid": candidate_edge_data["source_node_uuid"], # This is n1.uuid
            "target_node_uuid": candidate_edge_data["target_node_uuid"], # This is n2.uuid
            "score": candidate_edge_data["score"]
        }

        # The execute_query returns list of dicts.
        mock_kuzu_connection.prepare.return_value = mock_ps_invalid
        mock_kuzu_connection.execute.return_value = ([mock_result_dict], list(mock_result_dict.keys()), {})


        results_list = await kuzudb_provider.get_edge_invalidation_candidates(
            edges=[input_edge_ref],
            search_filter=EMPTY_SEARCH_FILTERS,
            min_score=min_score_threshold,
            limit=limit_val
        )

        assert len(results_list) == 1
        candidate_edges_for_ref = results_list[0]
        assert len(candidate_edges_for_ref) == 1
        assert candidate_edges_for_ref[0].uuid == "candidate-edge-uuid"

        # Check the query construction
        called_query = mock_kuzu_connection.prepare.call_args[0][0]
        called_params = mock_kuzu_connection.execute.call_args[1]

        assert "MATCH (n1:Entity)-[r_other:RELATES_TO]->(n2:Entity)" in called_query
        assert "(n1.uuid = $source_ref_uuid OR n1.uuid = $target_ref_uuid OR n2.uuid = $source_ref_uuid OR n2.uuid = $target_ref_uuid)" in called_query
        assert "r_other.uuid <> $edge_ref_uuid" in called_query
        assert "r_other.fact_embedding IS NOT NULL" in called_query
        assert "cosine_similarity(r_other.fact_embedding, $edge_ref_fact_embedding) AS score" in called_query
        assert "score >= $min_s" in called_query
        assert "ORDER BY score DESC LIMIT $lim" in called_query

        assert called_params["edge_ref_uuid"] == input_edge_ref.uuid
        assert called_params["edge_ref_fact_embedding"] == input_edge_ref.fact_embedding
        assert called_params["source_ref_uuid"] == input_edge_ref.source_node_uuid
        assert called_params["target_ref_uuid"] == input_edge_ref.target_node_uuid
        assert called_params["min_s"] == min_score_threshold
        assert called_params["lim"] == limit_val
        if input_edge_ref.group_id:
            assert "r_other.group_id = $edge_ref_group_id" in called_query
            assert called_params["edge_ref_group_id"] == input_edge_ref.group_id

    async def test_get_edge_invalidation_candidates_no_embedding(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        input_edge_ref_no_emb = EntityEdge(
            uuid="ref-edge-no-emb", name="REF_NO_EMB",
            source_node_uuid=MOCK_UUID_ENTITY_1, target_node_uuid=MOCK_UUID_ENTITY_2,
            fact="Fact no embedding", fact_embedding=None # No embedding
        )
        results_list = await kuzudb_provider.get_edge_invalidation_candidates(
            edges=[input_edge_ref_no_emb], search_filter=EMPTY_SEARCH_FILTERS, min_score=0.7, limit=5
        )
        assert len(results_list) == 1
        assert len(results_list[0]) == 0 # Should be empty as no embedding to search with
        mock_kuzu_connection.prepare.assert_not_called() # Query should not even be attempted

    # --- Error Handling Tests (Conceptual) ---
    async def test_get_entity_node_by_uuid_query_fails(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_kuzu_connection.prepare.side_effect = Exception("Kuzu prepare failed")
        # Or mock_kuzu_connection.execute.side_effect inside execute_query

        with pytest.raises(ConnectionError, match="KuzuDB query execution failed"): # Assuming ConnectionError is raised by execute_query
            await kuzudb_provider.get_entity_node_by_uuid("any-uuid")

    # --- Completeness for get_embeddings_for_... ---
    async def test_get_embeddings_for_communities(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        communities_in = [CommunityNode(uuid=MOCK_UUID_COMMUNITY_1, name="C1")]
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        cols = ["uuid", "embedding"]
        rows = [[MOCK_UUID_COMMUNITY_1, MOCK_KUZU_COMMUNITY_NODE_DICT["name_embedding"]]]
        configure_mock_query_result(mock_qr, cols, rows); mock_kuzu_connection.execute.return_value = mock_qr

        results = await kuzudb_provider.get_embeddings_for_communities(communities_in)
        assert len(results) == 1
        assert results[MOCK_UUID_COMMUNITY_1] == MOCK_KUZU_COMMUNITY_NODE_DICT["name_embedding"]
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "MATCH (c:Community) WHERE c.uuid IN $uuids RETURN c.uuid AS uuid, c.name_embedding AS embedding" in query_str

    async def test_get_embeddings_for_edges(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        edges_in = [EntityEdge(uuid=MOCK_UUID_EDGE_1, name="R1", source_node_uuid="s", target_node_uuid="t")]
        mock_ps = MagicMock(); mock_kuzu_connection.prepare.return_value = mock_ps; mock_qr = MagicMock()
        cols = ["uuid", "embedding"]
        rows = [[MOCK_UUID_EDGE_1, MOCK_KUZU_ENTITY_EDGE_DICT["fact_embedding"]]]
        configure_mock_query_result(mock_qr, cols, rows); mock_kuzu_connection.execute.return_value = mock_qr

        results = await kuzudb_provider.get_embeddings_for_edges(edges_in)
        assert len(results) == 1
        assert results[MOCK_UUID_EDGE_1] == MOCK_KUZU_ENTITY_EDGE_DICT["fact_embedding"]
        query_str = mock_kuzu_connection.prepare.call_args[0][0]
        assert "MATCH ()-[r:RELATES_TO]-() WHERE r.uuid IN $uuids RETURN r.uuid AS uuid, r.fact_embedding AS embedding" in query_str

```

[end of tests/providers/test_kuzudb_provider.py]
