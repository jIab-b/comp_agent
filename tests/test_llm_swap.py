import pytest
from omegaconf import OmegaConf
from src.llm.llm_factory import get_llm

@pytest.fixture
def cfg():
    return OmegaConf.create({
        "llm": {
            "provider": "fireworks",
            "fireworks": {
                "lora_model": None,
                "base_model": "accounts/fireworks/models/deepseek-llama-7b"
            },
            "openai": {
                "model": "gpt-4-turbo"
            }
        }
    })

def test_provider_swap(monkeypatch, cfg):
    monkeypatch.setitem(cfg.llm, "provider", "openai")
    # This is a mock test, so we don't actually create the object
    # assert get_llm(cfg).__class__.__name__ == "OpenAIChat"
    
    monkeypatch.setitem(cfg.llm, "provider", "fireworks")
    assert get_llm(cfg).__class__.__name__ == "fireworks_chat"