from langchain.agents import initialize_agent, AgentType
from omegaconf import DictConfig
from src.llm.llm_factory import get_llm
from src.core.interfaces import agent

class react_agent(agent):
    def __init__(self, cfg: DictConfig):
        self.llm = get_llm(cfg)
        self.tools = []
        self.memory = None

        self.agent_executor = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.REACT_DESCRIPTION,
            memory=self.memory,
            verbose=cfg.debug,
            max_iterations=cfg.agent.max_iterations,
            handle_parsing_errors=True,
        )

    def run(self, task: str) -> str:
        return self.agent_executor.run(task)