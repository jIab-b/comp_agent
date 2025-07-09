from omegaconf import DictConfig
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from src.llm.llm_factory import get_llm
from src.core.interfaces import agent
from src.agent.tool_loader import load_tools
from src.memory.memory_factory import get_memory

class react_agent(agent):
    def __init__(self, cfg: DictConfig):
        self.llm = get_llm(cfg)
        self.tools = load_tools(cfg)
        self.memory = get_memory(cfg)
        
        prompt = hub.pull("hwchase17/react")
        
        agent = create_react_agent(self.llm, self.tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=cfg.debug,
            max_iterations=cfg.agent.max_iterations,
            handle_parsing_errors=True,
        )

    def run(self, task: str) -> str:
        result = self.agent_executor.invoke({"input": task})
        return result["output"]