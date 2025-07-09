from omegaconf import DictConfig
from src.memory.vector_memory import get_vector_memory

def get_memory(cfg: DictConfig):
    backend = cfg.memory.backend
    if backend == "chroma":
        return get_vector_memory(cfg)
    else:
        raise ValueError(f"Unsupported memory backend: {backend}")