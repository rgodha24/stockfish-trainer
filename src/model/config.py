from dataclasses import dataclass

from .modules import LayerStacksConfig


# 3 layer fully connected network
@dataclass(kw_only=True)
class ModelConfig(LayerStacksConfig):
    pass
