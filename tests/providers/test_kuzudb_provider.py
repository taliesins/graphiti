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

    async def test_get_entity_nodes_by_group_ids(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes"]
        r1 = [MOCK_KUZU_ENTITY_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1]); mock_kuzu_connection.execute.return_value=mock_qr
        nodes = await kuzudb_provider.get_entity_nodes_by_group_ids([MOCK_GROUP_ID], limit=1)
        assert len(nodes) == 1 and nodes[0].uuid == MOCK_UUID_ENTITY_1
        assert "LIMIT $limit" in mock_kuzu_connection.prepare.call_args[0][0]

    async def test_get_entity_nodes_by_group_ids_no_limit(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["n.uuid", "n.name", "n.group_id", "n.created_at", "n.summary", "n.name_embedding", "n.attributes"]
        r1 = [MOCK_KUZU_ENTITY_NODE_DICT[k.split('.')[1]] for k in cols]; r2 = [MOCK_KUZU_ENTITY_NODE_DICT_2[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr, cols, [r1, r2]); mock_kuzu_connection.execute.return_value=mock_qr
        nodes = await kuzudb_provider.get_entity_nodes_by_group_ids([MOCK_GROUP_ID])
        assert len(nodes) == 2
        assert "LIMIT $limit" not in mock_kuzu_connection.prepare.call_args[0][0]

    # (Episodic & Community Node Save/Get/GetByGroup omitted for brevity but follow EntityNode structure)
    async def test_save_episodic_node(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        node = EpisodicNode(uuid=MOCK_UUID_EPISODIC_1, name="N1", group_id=MOCK_GROUP_ID, source=EpisodeType.text, entity_edges=[])
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock(); configure_mock_query_result(mock_qr,[],[], has_next=False); mock_kuzu_connection.execute.return_value=mock_qr
        await kuzudb_provider.save_episodic_node(node)
        assert "MERGE (n:Episodic {uuid: $match_uuid})" in mock_kuzu_connection.prepare.call_args[0][0]
        assert mock_kuzu_connection.execute.call_args[1]['p_source'] == EpisodeType.text.value

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

    async def test_get_entity_edge_by_uuid_found(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        mock_ps=MagicMock(); mock_kuzu_connection.prepare.return_value=mock_ps; mock_qr=MagicMock()
        cols = ["r.uuid", "r.name", "r.group_id", "r.fact", "r.fact_embedding", "r.episodes", "r.created_at", "r.expired_at", "r.valid_at", "r.invalid_at", "r.attributes", "source_node_uuid", "target_node_uuid"]
        data_dict = MOCK_KUZU_ENTITY_EDGE_DICT
        row_vals = [data_dict[k.split('.')[1] if '.' in k else k] for k in cols]
        configure_mock_query_result(mock_qr, cols, [row_vals]); mock_kuzu_connection.execute.return_value=mock_qr
        edge = await kuzudb_provider.get_entity_edge_by_uuid(MOCK_UUID_EDGE_1)
        assert edge is not None and edge.uuid == MOCK_UUID_EDGE_1

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
        # Note: community_fulltext_search doesn't take search_filter in current provider, so we pass EMPTY_SEARCH_FILTERS, but it's unused.
        mock_ps=MagicMock();mock_kuzu_connection.prepare.return_value=mock_ps;mock_qr=MagicMock()
        cols = ["c.uuid", "c.name", "c.group_id", "c.created_at", "c.summary", "c.name_embedding"]
        row1 = [MOCK_KUZU_COMMUNITY_NODE_DICT[k.split('.')[1]] for k in cols]
        configure_mock_query_result(mock_qr,cols,[row1]);mock_kuzu_connection.execute.return_value=mock_qr
        results = await kuzudb_provider.community_fulltext_search(query=search_query, group_ids=[MOCK_GROUP_ID], limit=mock_limit)
        assert len(results)==1 and results[0].uuid == MOCK_UUID_COMMUNITY_1
        assert "MATCH (c:Community)" in mock_kuzu_connection.prepare.call_args[0][0]
        assert "CONTAINS(lower(c.summary), $query_str_lower)" in mock_kuzu_connection.prepare.call_args[0][0]

    async def test_community_similarity_search(self, kuzudb_provider: KuzuDBProvider, mock_kuzu_connection: AsyncMock):
        search_vector=[.1,.2,.3]; mock_limit=1; min_score=0.7
        # Note: community_similarity_search doesn't take search_filter in current provider.
        mock_ps=MagicMock();mock_kuzu_connection.prepare.return_value=mock_ps;mock_qr=MagicMock()
        cols = ["c.uuid", "c.name", "c.group_id", "c.created_at", "c.summary", "c.name_embedding", "score"]
        row1 = [MOCK_KUZU_COMMUNITY_NODE_DICT[k.split('.')[1] if k != "score" else "uuid"] for k in cols]; row1[-1]=0.8
        configure_mock_query_result(mock_qr,cols,[row1]);mock_kuzu_connection.execute.return_value=mock_qr
        results = await kuzudb_provider.community_similarity_search(search_vector=search_vector, group_ids=[MOCK_GROUP_ID], limit=mock_limit, min_score=min_score)
        assert len(results)==1
        assert "cosine_similarity(c.name_embedding, $s_vec) >= $min_s" in mock_kuzu_connection.prepare.call_args[0][0]

```

[end of tests/providers/test_kuzudb_provider.py]
