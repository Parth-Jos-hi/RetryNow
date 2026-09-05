"""Simulator package: the SAFE execution layer (outcomes + latent oracle).

Nothing here touches real payment rails. Tests and demos execute against
:class:`PaymentSimulator`, which samples outcomes from the generator's latent
per-action truth and stays idempotent.
"""

from src.simulator.payment_simulator import (ALL_OUTCOMES, AttemptResult,
                                             PaymentSimulator)

__all__ = ["ALL_OUTCOMES", "AttemptResult", "PaymentSimulator"]