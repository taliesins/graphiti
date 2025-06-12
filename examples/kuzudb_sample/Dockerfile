FROM python:3.10-slim

WORKDIR /app

# Copy the main graphiti_core library into the Docker image
# This assumes the Docker build context is the root of the graphiti repository.
COPY ./graphiti_core /app/graphiti_core
COPY ./pyproject.toml /app/ # For poetry to identify the project
COPY ./poetry.lock /app/ # For poetry to identify the project


# Copy requirements for the sample app itself
COPY ./examples/kuzudb_sample/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# If graphiti_core itself needs to be "installed" in the image for its dependencies:
# This is a bit complex as graphiti-core uses poetry.
# A simpler way for an example is to ensure its path is added to PYTHONPATH.
ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY ./examples/kuzudb_sample/sample_app.py /app/

CMD ["python", "sample_app.py"]
