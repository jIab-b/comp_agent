import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
from src.agent.react_agent import react_agent
from dotenv import load_dotenv, find_dotenv
import os
import langchain

# Load environment variables
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

def main():
    """
    Runs a single chat prompt with the agent and exits.
    """
    parser = argparse.ArgumentParser(description="Run a single chat prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to send to the agent.")
    parser.add_argument("--lora", type=str, help="The name of the LoRA model to use.")
    parser.add_argument("--lang_debug", action="store_true", help="Enable langchain debug mode.")
    args = parser.parse_args()

    # Load base config
    cfg = OmegaConf.load(Path(__file__).parent.parent.parent / "config/settings.yaml")

    # Set langchain debug mode
    langchain.debug = args.lang_debug
    
    # Check for LoRA override
    if args.lora:
        try:
            model_id = load_lora_config(args.lora)
            print(f"--- Using LoRA model: {args.lora} ({model_id}) ---")
            cfg.llm.fireworks.model = model_id
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return

    # Initialize the agent and run the prompt
    agent = react_agent(cfg)
    result = agent.run(args.prompt)
    print(result)

if __name__ == "__main__":
    main()