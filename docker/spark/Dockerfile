# Spark base image with Delta Lake and Kafka connectors baked in.
# Used by spark-master, spark-worker, and the streaming job submitter.
FROM bitnamilegacy/spark:3.5.1

USER root

ARG SPARK_VERSION=3.5.1
ARG SCALA_VERSION=2.12
ARG DELTA_VERSION=3.2.0

# Base image does not ship curl; needed to fetch the connector jars.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Delta Lake (uber jar) + Kafka Structured Streaming jars (deterministic, no runtime download).
RUN mkdir -p /opt/bitnami/spark/jars && \
    curl -fsSL -o /opt/bitnami/spark/jars/delta-spark_${SCALA_VERSION}-${DELTA_VERSION}.jar \
      "https://repo1.maven.org/maven2/io/delta/delta-spark_${SCALA_VERSION}/${DELTA_VERSION}/delta-spark_${SCALA_VERSION}-${DELTA_VERSION}.jar" && \
    curl -fsSL -o /opt/bitnami/spark/jars/delta-storage-${DELTA_VERSION}.jar \
      "https://repo1.maven.org/maven2/io/delta/delta-storage/${DELTA_VERSION}/delta-storage-${DELTA_VERSION}.jar" && \
    curl -fsSL -o /opt/bitnami/spark/jars/spark-sql-kafka-0-10_${SCALA_VERSION}-${SPARK_VERSION}.jar \
      "https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_${SCALA_VERSION}/${SPARK_VERSION}/spark-sql-kafka-0-10_${SCALA_VERSION}-${SPARK_VERSION}.jar" && \
    curl -fsSL -o /opt/bitnami/spark/jars/spark-token-provider-kafka-0-10_${SCALA_VERSION}-${SPARK_VERSION}.jar \
      "https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_${SCALA_VERSION}/${SPARK_VERSION}/spark-token-provider-kafka-0-10_${SCALA_VERSION}-${SPARK_VERSION}.jar" && \
    curl -fsSL -o /opt/bitnami/spark/jars/kafka-clients-${SPARK_VERSION}.jar \
      "https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/${SPARK_VERSION}/kafka-clients-${SPARK_VERSION}.jar" && \
    curl -fsSL -o /opt/bitnami/spark/jars/commons-pool2-2.12.0.jar \
      "https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.12.0/commons-pool2-2.12.0.jar"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The Delta volume mount point must be writable by the non-root spark user (uid 1001).
RUN mkdir -p /opt/spark/work-dir/delta && chown -R 1001:0 /opt/spark/work-dir/delta

USER 1001
