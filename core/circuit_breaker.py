"""
PATO QUANT — Circuit Breaker para APIs externas
Evita saturar APIs que fallan repetidamente (yfinance, Groq, Alpaca).

Lógica:
  - Después de N fallos consecutivos, el circuito se "abre" y bloquea
    llamadas durante un período de cooldown.
  - Después del cooldown, permite UNA llamada de prueba (half-open).
  - Si la prueba pasa, el circuito se cierra de nuevo.
  - Si falla, se reabre por otro período de cooldown.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger("PatoQuant.CircuitBreaker")


class CircuitBreaker:
    """Circuit breaker para proteger contra APIs caídas."""

    CLOSED = "CLOSED"       # Normal — permite llamadas
    OPEN = "OPEN"           # Bloqueado — rechaza llamadas
    HALF_OPEN = "HALF_OPEN" # Probando — permite 1 llamada

    def __init__(self, name: str,
                 failure_threshold: int = 5,
                 cooldown_seconds: int = 120):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None

    def can_execute(self) -> bool:
        """Retorna True si se puede ejecutar la llamada."""
        if self.state == self.CLOSED:
            return True

        if self.state == self.OPEN:
            # ¿Ya pasó el cooldown?
            if (self.last_failure_time and
                    time.time() - self.last_failure_time >= self.cooldown_seconds):
                self.state = self.HALF_OPEN
                logger.info(f"🔄 Circuit breaker [{self.name}]: HALF_OPEN (probando)")
                return True
            return False

        # HALF_OPEN: permitir 1 llamada
        return True

    def record_success(self):
        """Registra éxito — resetea el circuito."""
        if self.state == self.HALF_OPEN:
            logger.info(f"✅ Circuit breaker [{self.name}]: CLOSED (recuperado)")
        self.failure_count = 0
        self.state = self.CLOSED

    def record_failure(self):
        """Registra fallo — puede abrir el circuito."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN
            logger.warning(
                f"🔴 Circuit breaker [{self.name}]: OPEN "
                f"({self.failure_count} fallos, cooldown {self.cooldown_seconds}s)"
            )

    def get_status(self) -> str:
        return f"{self.name}: {self.state} (fallos: {self.failure_count})"
