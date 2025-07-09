import hydra
from omegaconf import DictConfig
from src.agent.react_agent import react_agent
from dotenv import load_dotenv, find_dotenv
import os
import langchain
load_dotenv(find_dotenv('.env.local'))

@hydra.main(config_path="../../config", config_name="settings", version_base=None)
def main(cfg: DictConfig) -> None:
    # Set langchain debug mode based on the command-line argument
    langchain.debug = cfg.get('lang_debug', False)
    print("--- Initializing Agent ---")
    agent = react_agent(cfg)
    print("--- Agent Initialized. Type 'exit' or 'quit' to end the session. ---")

    while True:
        try:
            task = input("\nYou: ")
            if task.lower() in ["exit", "quit"]:
                print("\n--- Session Ended ---")
                break
            
            result = agent.run(task)
            print(f"\nAssistant: {result}")

        except KeyboardInterrupt:
            print("\n--- Session Ended ---")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()