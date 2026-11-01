from .sequence import SamplingParams, Sequence


def __getattr__(name):            # lazy so the pure-Python parts import without torch
    if name == "LLMEngine":
        from .engine import LLMEngine
        return LLMEngine
    raise AttributeError(name)


__all__ = ["LLMEngine", "SamplingParams", "Sequence"]
