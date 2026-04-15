from collections.abc import Callable
from .composed import ComposedFeatureTransformer, combine_input_features
from .full_threats import FullThreats
from .halfka_v2_hm import HalfKav2Hm
from .input_feature import InputFeature

_FEATURE_COMPONENTS: dict[str, type[InputFeature]] = {
    "HalfKAv2_hm^": HalfKav2Hm,
    "Full_Threats": FullThreats,
}


def get_feature_cls(name: str) -> Callable[[int], ComposedFeatureTransformer]:
    parts = name.split("+")
    components = [_FEATURE_COMPONENTS[p] for p in parts]
    return combine_input_features(*components)


__all__ = [
    "ComposedFeatureTransformer",
    "combine_input_features",
    "HalfKav2Hm",
    "FullThreats",
    "InputFeature",
    "get_feature_cls",
]
