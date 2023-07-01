from ...constants import Direction
from ..._utils import negate_bounds as _not
from ..activations.neuron.static import _StaticActivation

import torch


class Godel(_StaticActivation):
    """Weighted Godel"""

    def _and_upward(self, operand_bounds: torch.Tensor):
        # print('-'*100)
        # print(operand_bounds.shape, self.weights.shape)
        # print('-'*100)
        return (self.bias - torch.max((1 - operand_bounds) * torch.broadcast_to(self.weights, operand_bounds.shape),-1).values).clamp(0, 1)

    
    def _or_upward(self, operand_bounds: torch.Tensor):
        return (1-self.bias + torch.max((1 - operand_bounds) * torch.broadcast_to(self.weights, operand_bounds.shape),-1).values).clamp(0, 1)
