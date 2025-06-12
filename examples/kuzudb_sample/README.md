# Graphiti KuzuDB Provider Sample Project

This sample project demonstrates basic usage of the `KuzuDBProvider` from the `graphiti-core` library to interact with a KuzuDB graph database.

## Prerequisites

*   Docker
*   Docker Compose

## Overview

The sample application (`sample_app.py`):
1.  Connects to a KuzuDB instance (Kuzu is used as an embedded library within the app's Docker container).
2.  Builds the graph schema (tables for nodes and relationships), deleting any existing schema/data first.
3.  Saves a few `EntityNode`s and `EpisodicNode`s.
4.  Saves `EntityEdge`s (relationships between entities) and `EpisodicEdge`s (mentions).
5.  Demonstrates a bulk operation for adding nodes and edges, which also handles generating dummy embeddings for items in the bulk call.
6.  Retrieves a node by its UUID to show successful persistence and embedding generation.
7.  Performs a full-text search for nodes.
8.  Performs a vector similarity search for nodes using the generated dummy embeddings.

KuzuDB data created by the `sample_app` is persisted in a Docker named volume (`app_kuzu_data_vol`), which is mounted into the `sample_app` container at `/app/kuzu_db_volume/`.

The `kuzudb_service` defined in `docker-compose.yml` is largely a placeholder for this example, as KuzuDB is used in its embedded mode directly by the `sample_app`. If KuzuDB had a standalone server mode that was being used, that service definition would be more critical.

## Running the Sample

1.  **Navigate to this directory:**
    ```bash
    cd examples/kuzudb_sample
    ```

2.  **Build and run the services using Docker Compose:**
    Make sure you are in the `examples/kuzudb_sample` directory. The Docker build context is set to `../../` (the root of the `graphiti` repository) to allow copying `graphiti_core`.
    ```bash
    docker-compose up --build
    ```
    This command will:
    *   Build the Docker image for the `sample_app` service (which includes `graphiti_core` and its dependencies).
    *   Pull the latest KuzuDB Docker image (though not actively used as a server by `sample_app`).
    *   Start the KuzuDB service (placeholder) and the sample application service.

3.  **View Output:**
    The `sample_app.py` script will log its operations to the console. You should see logs indicating connection, schema building, data insertion, embedding generation, and query results.

4.  **Shutdown:**
    Press `Ctrl+C` in the terminal where `docker-compose up` is running. To remove the containers and network:
    ```bash
    docker-compose down
    ```
    To also remove the persisted KuzuDB data volume (`app_kuzu_data_vol`):
    ```bash
    docker-compose down -v
    ```

## Project Structure

*   `sample_app.py`: The Python application logic using `KuzuDBProvider`.
*   `docker-compose.yml`: Defines the (placeholder) KuzuDB and application services.
*   `Dockerfile`: Instructions to build the Docker image for `sample_app.py`, including `graphiti_core`.
*   `requirements.txt`: Python dependencies for the `sample_app.py`.
*   `README.md`: This file.
*   `kuzu_sample_data/`: Directory intended for the `kuzudb_service` if it were persisting data (less relevant for this embedded sample).
*   The actual KuzuDB database file created by `sample_app.py` is stored in a Docker named volume (`app_kuzu_data_vol`) and mapped to `/app/kuzu_db_volume/` inside the `sample_app` container. You won't see this as a local directory unless you change the volume mapping in `docker-compose.yml` to a local bind mount.
