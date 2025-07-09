from omegaconf import DictConfig
from src.core.interfaces import llm
from src.llm.fireworks_provider import fireworks_chat

def get_llm(cfg: DictConfig) -> llm:
    provider = cfg.llm.provider
    
    if provider == "fireworks":
        # Pass the entire fireworks config to the constructor
        return fireworks_chat(**cfg.llm.fireworks)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")