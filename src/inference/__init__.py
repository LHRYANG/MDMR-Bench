__all__ = [
    "BaseInferencer",
    "Flux2Inferencer",
    "QwenImageEditInferencer",
    "get_inferencer_class",
]

def __getattr__(name):
    if name == "BaseInferencer":
        from .base import BaseInferencer
        return BaseInferencer
    if name == "Flux2Inferencer":
        from .flux2 import Flux2Inferencer
        return Flux2Inferencer
    if name == "QwenImageEditInferencer":
        from .qwen_image_edit import QwenImageEditInferencer
        return QwenImageEditInferencer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_inferencer_class(name):
    name_lower = name.lower()
    
    if "qwen" in name_lower:
        from .qwen_image_edit import QwenImageEditInferencer
        return QwenImageEditInferencer
    elif "flux" in name_lower and "2" in name_lower:
        from .flux2 import Flux2Inferencer
        return Flux2Inferencer

    return None
