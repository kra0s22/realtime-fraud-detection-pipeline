# Dev-only image to run the pytest suite without a local Python environment.
# Inherits the built Spark image (pyspark, delta-spark, redis, pydantic) and
# adds the test-only dependencies on top.
FROM fraud-pipeline-spark-master:latest

USER root

# git is required by GitPython, which MLflow uses to record the source commit.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY docker/test/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

USER 1001
