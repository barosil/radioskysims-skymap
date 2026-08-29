"""
Telemetria estruturada para adapters de SkyMap.

Fornece:
- Medição de tempo de execução
- Registro de eventos-chave
- Logging estruturado
- Fácil de ativar/desativar

Uso:
    from radioskysims_skymap.infrastructure.skymap.telemetry import Telemetry

    telemetry = Telemetry(__name__)

    with telemetry.measure("operation_name"):
        # código a medir
        result = operation()

    telemetry.event("key_event", extra_data={"param": value})
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

# Configuração de telemetria global
_TELEMETRY_ENABLED = True


def enable_telemetry() -> None:
    """Ativa telemetria globalmente."""
    global _TELEMETRY_ENABLED
    _TELEMETRY_ENABLED = True


def disable_telemetry() -> None:
    """Desativa telemetria globalmente."""
    global _TELEMETRY_ENABLED
    _TELEMETRY_ENABLED = False


def is_telemetry_enabled() -> bool:
    """Verifica se telemetria está ativa."""
    return _TELEMETRY_ENABLED


@dataclass
class TimingRecord:
    """Registro de medição de tempo."""

    operation: str
    elapsed_ms: float
    success: bool
    error: str | None = None


@dataclass
class EventRecord:
    """Registro de evento."""

    event_name: str
    timestamp: float
    metadata: dict[str, Any]


class Telemetry:
    """
    Telemetria estruturada para operações em adapters.

    Exemplo:
        telemetry = Telemetry(__name__)

        # Medir tempo
        with telemetry.measure("compute_cls"):
            result = hp.anafast(data)

        # Registrar evento
        telemetry.event("cls_computed", size=len(result))

        # Obter estatísticas
        stats = telemetry.stats()
    """

    def __init__(self, name: str):
        """
        Inicializa telemetria.

        Args:
            name: Nome do módulo/classe (ex: __name__)
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self._timings: list[TimingRecord] = []
        self._events: list[EventRecord] = []

    @contextmanager
    def measure(self, operation: str) -> Iterator[None]:
        """
        Context manager para medir tempo de execução.

        Args:
            operation: Nome da operação

        Yields:
            None

        Example:
            with telemetry.measure("expensive_operation"):
                result = expensive_function()
        """
        if not _TELEMETRY_ENABLED:
            yield
            return

        start = time.perf_counter()
        success = False
        error_msg = None

        try:
            yield
            success = True
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            elapsed = time.perf_counter() - start
            elapsed_ms = elapsed * 1000

            # Registra medição
            record = TimingRecord(
                operation=operation,
                elapsed_ms=elapsed_ms,
                success=success,
                error=error_msg,
            )
            self._timings.append(record)

            # Log estruturado
            if success:
                self.logger.debug(
                    f"⏱️  {operation}: {elapsed_ms:.2f}ms",
                    extra={
                        "operation": operation,
                        "elapsed_ms": elapsed_ms,
                        "success": True,
                    },
                )
            else:
                self.logger.warning(
                    f"❌ {operation} FAILED: {error_msg} ({elapsed_ms:.2f}ms)",
                    extra={
                        "operation": operation,
                        "elapsed_ms": elapsed_ms,
                        "success": False,
                        "error": error_msg,
                    },
                )

    def event(self, event_name: str, **metadata: Any) -> None:
        """
        Registra evento com metadados opcionais.

        Args:
            event_name: Nome do evento
            **metadata: Dados adicionais

        Example:
            telemetry.event("map_loaded", size=12, nside=1)
        """
        if not _TELEMETRY_ENABLED:
            return

        import time

        record = EventRecord(
            event_name=event_name,
            timestamp=time.time(),
            metadata=metadata,
        )
        self._events.append(record)

        # Log com metadados
        if metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
            self.logger.info(f"📌 {event_name}: {meta_str}", extra=metadata)
        else:
            self.logger.info(f"📌 {event_name}")

    def stats(self) -> dict[str, Any]:
        """
        Retorna estatísticas de telemetria.

        Returns:
            Dicionário com estatísticas agregadas
        """
        if not self._timings:
            return {
                "operations": 0,
                "total_time_ms": 0.0,
                "events": len(self._events),
            }

        successful = [t for t in self._timings if t.success]
        failed = [t for t in self._timings if not t.success]

        total_ms = sum(t.elapsed_ms for t in self._timings)
        avg_ms = total_ms / len(self._timings) if self._timings else 0

        return {
            "operations": len(self._timings),
            "successful": len(successful),
            "failed": len(failed),
            "total_time_ms": total_ms,
            "avg_time_ms": avg_ms,
            "min_time_ms": min(t.elapsed_ms for t in self._timings),
            "max_time_ms": max(t.elapsed_ms for t in self._timings),
            "events": len(self._events),
        }

    def clear(self) -> None:
        """Limpa registros de telemetria."""
        self._timings.clear()
        self._events.clear()

    def dump(self) -> dict[str, Any]:
        """
        Retorna dump completo de telemetria (para análise).

        Returns:
            Dicionário com timings e eventos detalhados
        """
        return {
            "module": self.name,
            "enabled": _TELEMETRY_ENABLED,
            "timings": [
                {
                    "operation": t.operation,
                    "elapsed_ms": t.elapsed_ms,
                    "success": t.success,
                    "error": t.error,
                }
                for t in self._timings
            ],
            "events": [
                {
                    "event": e.event_name,
                    "timestamp": e.timestamp,
                    "metadata": e.metadata,
                }
                for e in self._events
            ],
        }
