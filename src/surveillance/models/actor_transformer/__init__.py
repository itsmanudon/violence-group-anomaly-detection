"""Actor-Transformer for annotated-actor group activity recognition."""

from surveillance.models.actor_transformer.actor_transformer import ActorTransformer
from surveillance.models.actor_transformer.loss import ActorGroupLoss

__all__ = ["ActorTransformer", "ActorGroupLoss"]
