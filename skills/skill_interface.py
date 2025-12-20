from collections import namedtuple
from torch import Tensor

# Standard Skill tuple used across the project
Skill = namedtuple('Skill', ['name', 'input_adapter', 'skill_model', 'skill_output', 'skill_adapter'])


def model_forward(model, x):
    """Default forward wrapper for skill models."""
    return model(x)


def identity_adapter(x: Tensor) -> Tensor:
    """A simple adapter that returns the input unchanged."""
    return x
