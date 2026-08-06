"""Unit tests for the producer loop using a fake Kafka producer."""

from producer.config import ProducerSettings
from producer.generator import TransactionGenerator
from producer.main import run
from producer.models import Transaction


class FakeProducer:
    def __init__(self):
        self.produced = []
        self.flushed = 0
        self.polled = 0

    def produce(self, topic, key, value, callback=None):
        self.produced.append((topic, key, value, callback))

    def poll(self, timeout=0):
        self.polled += 1
        return 0

    def flush(self, timeout=None):
        self.flushed += 1


def _settings(**overrides):
    defaults = {
        "bootstrap_servers": "localhost:9092",
        "transactions_per_second": 1000.0,
    }
    defaults.update(overrides)
    return ProducerSettings(**defaults)


def test_run_produces_expected_count():
    settings = _settings(max_transactions=5)
    producer = FakeProducer()
    count = run(settings, TransactionGenerator(seed=1), producer)

    assert count == 5
    assert len(producer.produced) == 5
    assert producer.flushed == 1
    assert all(topic == "transactions.raw" for topic, *_ in producer.produced)
    for _, key, value, _ in producer.produced:
        tx = Transaction.model_validate_json(value)
        assert tx.transaction_id == key


def test_run_flushes_on_interrupt():
    class InterruptingGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self):
            self.calls += 1
            if self.calls == 3:
                raise KeyboardInterrupt
            return TransactionGenerator(seed=1).generate()

    settings = _settings()
    producer = FakeProducer()
    count = run(settings, InterruptingGenerator(), producer)

    assert count == 2
    assert len(producer.produced) == 2
    assert producer.flushed == 1


def test_run_recovers_from_buffer_error():
    class FlakyProducer(FakeProducer):
        def __init__(self):
            super().__init__()
            self.raise_buffer = True

        def produce(self, topic, key, value, callback=None):
            if self.raise_buffer:
                self.raise_buffer = False
                raise BufferError
            super().produce(topic, key, value, callback)

    settings = _settings(max_transactions=3)
    producer = FlakyProducer()
    count = run(settings, TransactionGenerator(seed=1), producer)

    assert count == 3
    assert len(producer.produced) == 3
    assert producer.flushed == 1
