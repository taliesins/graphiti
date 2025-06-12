import asyncio
import os
import logging
from datetime import datetime, timezone

# Assuming graphiti_core is in PYTHONPATH (handled by Dockerfile)
from graphiti_core.providers.kuzudb_provider import KuzuDBProvider, SearchFilters
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder.client import EmbedderClient # Mock embedder

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock Embedder for the sample
class MockEmbedder(EmbedderClient):
    async def create(self, input_data: list[str]) -> list[list[float]]:
        logger.info(f"MockEmbedder: Generating dummy embeddings for {len(input_data)} items.")
        # Return 10-dimensional embeddings as specified in KuzuDBProvider init
        return [[float(i+1)/10.0] * 10 for i in range(len(input_data))]

async def main():
    # KUZU_DB_PATH is defined in docker-compose.yml for the sample_app service
    kuzu_db_file_path = os.environ.get("KUZU_DB_PATH", "/tmp/kuzu_sample_db/sample_database.kuzu")
    # Ensure the directory for Kuzu DB exists
    kuzu_db_dir = os.path.dirname(kuzu_db_file_path)
    os.makedirs(kuzu_db_dir, exist_ok=True)

    logger.info(f"Initializing KuzuDBProvider with database path: {kuzu_db_file_path}")
    # embedding_dimension must match MockEmbedder output
    provider = KuzuDBProvider(database_path=kuzu_db_file_path, in_memory=False, embedding_dimension=10)
    mock_embedder = MockEmbedder()

    try:
        logger.info("Connecting to KuzuDB...")
        await provider.connect()
        logger.info("Connection successful.")

        logger.info("Building schema (indices and constraints)...")
        # Delete existing data and schema for a clean run each time
        await provider.build_indices_and_constraints(delete_existing=True)
        logger.info("Schema build complete.")

        # Create some nodes
        entity1 = EntityNode(uuid="entity-1", name="Alice Wonderland", group_id="sample_group", summary="Software Engineer curious about graphs.")
        entity2 = EntityNode(uuid="entity-2", name="Bob The Builder", group_id="sample_group", summary="Data Scientist skilled in graph algorithms.")

        episodic1 = EpisodicNode(
            uuid="episode-1", name="Project Alpha Kickoff Meeting", group_id="sample_group",
            source=EpisodeType.meeting, content="Discussed project milestones for Alpha and KuzuDB integration.",
            valid_at=datetime.now(timezone.utc)
        )

        logger.info("Saving nodes (will trigger embedding generation via bulk add later)...")
        # Note: Standalone save methods don't typically call embedders in graphiti-core by default.
        # Embeddings are usually handled by higher-level logic or bulk operations.
        # For this sample, we'll rely on add_nodes_and_edges_bulk to generate embeddings.
        # To ensure nodes have embeddings for individual save, we'd mock/call embedder before save.
        # However, KuzuDBProvider's add_nodes_and_edges_bulk DOES handle embedding generation.

        # For simplicity in sample, let's ensure embeddings are present if we want to demo similarity search soon.
        # Option 1: Call embedder explicitly (not typical for provider direct use)
        # Option 2: Rely on bulk add (done later)
        # Option 3: Save nodes without embedding then update (more complex for sample)

        # Let's save them, and then use bulk add for a node that will get an embedding.
        await provider.save_entity_node(entity1)
        await provider.save_entity_node(entity2)
        await provider.save_episodic_node(episodic1)
        logger.info("Initial nodes saved (without explicit embedding generation yet).")

        # Create some edges
        edge1 = EntityEdge(
            uuid="edge-1", name="COLLEAGUES", source_node_uuid=entity1.uuid, target_node_uuid=entity2.uuid,
            fact="Alice and Bob are colleagues working on KuzuDB samples.", group_id="sample_group"
        )
        ep_edge1 = EpisodicEdge(
            uuid="ep-edge-1", source_node_uuid=episodic1.uuid, target_node_uuid=entity1.uuid,
            group_id="sample_group"
        )

        logger.info("Saving edges...")
        await provider.save_entity_edge(edge1)
        await provider.save_episodic_edge(ep_edge1)
        logger.info("Edges saved.")

        logger.info("Performing a bulk operation to add nodes/edges with embedding generation...")
        entity_bulk_charlie = EntityNode(uuid="entity-charlie", name="Charlie Brown", group_id="sample_group", summary="Project Manager overseeing graph projects.")
        # Edge for Charlie, fact will be embedded
        edge_bulk_charlie_manages = EntityEdge(
            uuid="edge-charlie-manages", name="MANAGES", source_node_uuid=entity_bulk_charlie.uuid, target_node_uuid=entity1.uuid,
            fact="Charlie Brown manages Alice Wonderland on Project Alpha.", group_id="sample_group"
        )
        # Update Alice to ensure her embedding is generated via bulk op if not already
        # For KuzuDB, save_entity_node is an upsert. If Alice already exists, it updates.
        # The bulk operation will also upsert.
        # To ensure Alice gets an embedding, let's include her in the bulk operation.
        # This assumes the bulk operation will generate embedding for existing nodes if missing.
        # The KuzuDB provider's bulk method generates embeddings for nodes in the entity_nodes list passed to it.

        await provider.add_nodes_and_edges_bulk(
            episodic_nodes=[],
            episodic_edges=[],
            entity_nodes=[entity1, entity_bulk_charlie], # Alice (entity1) included to ensure her embedding is generated
            entity_edges=[edge_bulk_charlie_manages],
            embedder=mock_embedder
        )
        logger.info("Bulk operation complete. Embeddings should now be generated for Alice and Charlie, and Charlie's edge fact.")

        logger.info(f"Retrieving node '{entity1.name}' by UUID...")
        retrieved_alice = await provider.get_entity_node_by_uuid(entity1.uuid)
        if retrieved_alice:
            logger.info(f"Found Alice: {retrieved_alice.name}, Summary: {retrieved_alice.summary}, Embedding generated: {retrieved_alice.name_embedding is not None}")
        else:
            logger.error("Alice not found!")

        logger.info("Performing a full-text search for nodes with 'Data Scientist'...")
        search_results_ds = await provider.node_fulltext_search(query="Data Scientist", search_filter=SearchFilters(), group_ids=["sample_group"])
        if search_results_ds:
            logger.info(f"Found {len(search_results_ds)} node(s) via full-text search for 'Data Scientist':")
            for node in search_results_ds:
                logger.info(f"  - {node.name} (UUID: {node.uuid})")
        else:
            logger.info("No nodes found via full-text search for 'Data Scientist'.")

        logger.info("Performing a similarity search for nodes similar to Alice (embedding should exist now)...")
        if retrieved_alice and retrieved_alice.name_embedding:
            similar_nodes = await provider.node_similarity_search(
                search_vector=retrieved_alice.name_embedding,
                search_filter=SearchFilters(),
                group_ids=["sample_group"],
                limit=3 # Increase limit to see more potential results
            )
            if similar_nodes:
                logger.info(f"Found {len(similar_nodes)} node(s) potentially similar to Alice (including Alice):")
                for node in similar_nodes:
                    logger.info(f"  - {node.name} (UUID: {node.uuid}), Has embedding: {node.name_embedding is not None}")
            else:
                logger.info("No similar nodes found for Alice.")
        else:
            logger.info("Alice not retrieved or no embedding for similarity search.")

        logger.info("Sample operations complete.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if provider:
            logger.info("Closing KuzuDB connection.")
            await provider.close()
            logger.info("Connection closed.")

if __name__ == "__main__":
    asyncio.run(main())
