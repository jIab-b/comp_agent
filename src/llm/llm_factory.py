from omegaconf import DictConfig
from src.core.interfaces import llm
from src.llm.fireworks_provider import fireworks_chat

def get_llm(cfg: DictConfig) -> llm:
    provider = cfg.llm.provider
    
    if provider == "fireworks":
        model = cfg.llm.fireworks.lora_model or cfg.llm.fireworks.base_model
        return fireworks_chat(model=model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")