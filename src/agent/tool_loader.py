from omegaconf import DictConfig
from src.tools.web_search import get_web_search_tool

def load_tools(cfg: DictConfig):
    tools = []
    if "web_search" in cfg.agent.tools:
        tools.append(get_web_search_tool())
    return tools