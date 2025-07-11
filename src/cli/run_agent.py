import hydra
import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from src.agent.react_agent import react_agent
from dotenv import load_dotenv, find_dotenv
import os
import langchain
load_dotenv(find_dotenv('.env.local'))

def load_lora_config(lora_name: str) -> str:
    """Loads the model ID for a given LoRA from the registry."""
    registry_path = Path(__file__).parent.parent / "lora/registry.json"
    if not registry_path.exists():
        raise FileNotFoundError("LoRA registry not found.")
        
    with registry_path.open("r") as f:
        registry = json.load(f)
        
    for entry in registry:
        if entry["name"] == lora_name:
            return entry["model_id"]
            
    raise ValueError(f"LoRA '{lora_name}' not found in the registry.")

@hydra.main(config_path="../../config", config_name="settings", version_base=None)
def main(cfg: DictConfig) -> None:
    # Set langchain debug mode based on the command-line argument
    langchain.debug = cfg.get('lang_debug', False)
    
    # Check for LoRA override
    lora_name = cfg.get("lora")
    if lora_name:
        try:
            model_id = load_lora_config(lora_name)
            print(f"--- Using LoRA model: {lora_name} ({model_id}) ---")
            OmegaConf.set_struct(cfg, False)
            cfg.llm.fireworks.model = model_id
            OmegaConf.set_struct(cfg, True)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return

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