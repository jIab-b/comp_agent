import hydra
from omegaconf import DictConfig
from src.agent.react_agent import react_agent

@hydra.main(config_path="../../config", config_name="settings", version_base=None)
def main(cfg: DictConfig) -> None:
    task = "Draft a blog post on modular AI agents"
    print(f"--- Running Agent on Task: {task} ---")
    
    agent = react_agent(cfg)
    result = agent.run(task)
    
    print(f"--- Agent Finished ---")
    print(result)

if __name__ == "__main__":
    main()