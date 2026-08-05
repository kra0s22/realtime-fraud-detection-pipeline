"""Entry point for the synthetic transaction producer."""

import logging
import time

from confluent_kafka import Producer

from producer.config import ProducerSettings
from producer.generator import TransactionGenerator

logger = logging.getLogger("producer")


def _delivery_callback(err, msg) -> None:
    if err is not None:
        logger.error("Delivery failed for %s: %s", msg.topic(), err)


def run(settings: ProducerSettings, generator: TransactionGenerator, producer) -> int:
    """Produce transactions until max_transactions is reached or the process is interrupted."""
    produced = 0
    interval = 1.0 / settings.transactions_per_second
    try:
        while settings.max_transactions is None or produced < settings.max_transactions:
            tx = generator.generate()
            try:
                producer.produce(
                    settings.topic,
                    key=tx.transaction_id,
                    value=tx.model_dump_json(),
                    callback=_delivery_callback,
                )
            except BufferError:
                # Local queue is full; drain delivery reports before retrying.
                producer.poll(1)
                producer.produce(
                    settings.topic,
                    key=tx.transaction_id,
                    value=tx.model_dump_json(),
                    callback=_delivery_callback,
                )
            produced += 1
            producer.poll(0)
            if settings.max_transactions is not None and produced >= settings.max_transactions:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Interrupted; flushing pending messages.")
    finally:
        producer.flush(10)
    return produced


def main() -> None:
    settings = ProducerSettings()
    generator = TransactionGenerator(fraud_rate=settings.fraud_rate, seed=settings.seed)
    conf = {
        "bootstrap.servers": settings.bootstrap_servers,
        "client.id": "fraud-producer",
        "acks": "all",
    }
    producer = Producer(conf)
    count = run(settings, generator, producer)
    logger.info("Produced %d transaction(s) to '%s'", count, settings.topic)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
