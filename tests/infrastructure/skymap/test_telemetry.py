"""
Testes para o sistema de telemetria de adapters.
"""

import pytest

from radioskysims_skymap.infrastructure.skymap.telemetry import (
    Telemetry,
    enable_telemetry,
    disable_telemetry,
    is_telemetry_enabled,
)


class TestTelemetryBasics:
    """Testes para funcionalidades básicas de telemetria."""

    def test_telemetry_enable_disable(self):
        """Testa ativar/desativar telemetria."""
        assert is_telemetry_enabled()

        disable_telemetry()
        assert not is_telemetry_enabled()

        enable_telemetry()
        assert is_telemetry_enabled()

    def test_telemetry_measure(self):
        """Testa medição de tempo."""
        telemetry = Telemetry("test_module")

        with telemetry.measure("operation"):
            result = 1 + 1

        assert result == 2
        assert len(telemetry._timings) == 1
        assert telemetry._timings[0].operation == "operation"
        assert telemetry._timings[0].success
        assert telemetry._timings[0].elapsed_ms > 0

    def test_telemetry_measure_disabled(self):
        """Testa que measure não registra quando desativado."""
        disable_telemetry()
        try:
            telemetry = Telemetry("test_module")

            with telemetry.measure("operation"):
                pass

            # Não deve ter registrado nada
            assert len(telemetry._timings) == 0
        finally:
            enable_telemetry()

    def test_telemetry_measure_exception(self):
        """Testa que exception é registrada."""
        telemetry = Telemetry("test_module")

        with pytest.raises(ValueError):
            with telemetry.measure("failing_operation"):
                msg = "Test error"
                raise ValueError(msg)

        assert len(telemetry._timings) == 1
        assert not telemetry._timings[0].success
        assert "Test error" in telemetry._timings[0].error

    def test_telemetry_event(self):
        """Testa registros de evento."""
        telemetry = Telemetry("test_module")

        telemetry.event("my_event", param1="value1", param2=42)

        assert len(telemetry._events) == 1
        assert telemetry._events[0].event_name == "my_event"
        assert telemetry._events[0].metadata["param1"] == "value1"
        assert telemetry._events[0].metadata["param2"] == 42

    def test_telemetry_event_disabled(self):
        """Testa que event não registra quando desativado."""
        disable_telemetry()
        try:
            telemetry = Telemetry("test_module")
            telemetry.event("my_event", data="value")

            assert len(telemetry._events) == 0
        finally:
            enable_telemetry()

    def test_telemetry_stats(self):
        """Testa coleta de estatísticas."""
        telemetry = Telemetry("test_module")

        with telemetry.measure("op1"):
            pass

        with telemetry.measure("op2"):
            pass

        telemetry.event("event1")
        telemetry.event("event2")

        stats = telemetry.stats()

        assert stats["operations"] == 2
        assert stats["successful"] == 2
        assert stats["failed"] == 0
        assert stats["events"] == 2
        assert stats["avg_time_ms"] > 0

    def test_telemetry_clear(self):
        """Testa limpeza de telemetria."""
        telemetry = Telemetry("test_module")

        with telemetry.measure("operation"):
            pass

        telemetry.event("event")

        assert telemetry.stats()["operations"] == 1
        assert telemetry.stats()["events"] == 1

        telemetry.clear()

        assert telemetry.stats()["operations"] == 0
        assert telemetry.stats()["events"] == 0

    def test_telemetry_dump(self):
        """Testa dump completo."""
        telemetry = Telemetry("test_module")

        with telemetry.measure("op"):
            pass

        telemetry.event("evt", data="test")

        dump = telemetry.dump()

        assert dump["module"] == "test_module"
        assert dump["enabled"] is True
        assert len(dump["timings"]) == 1
        assert len(dump["events"]) == 1
        assert dump["timings"][0]["operation"] == "op"
        assert dump["events"][0]["event"] == "evt"

    def test_telemetry_multiple_operations(self):
        """Testa múltiplas operações."""
        telemetry = Telemetry("test_module")

        for i in range(5):
            with telemetry.measure(f"operation_{i}"):
                pass

        stats = telemetry.stats()
        assert stats["operations"] == 5
        assert stats["successful"] == 5
        assert stats["min_time_ms"] > 0
        assert stats["max_time_ms"] >= stats["min_time_ms"]
