import time
import json
from pathlib import Path
from datetime import datetime
from .client import fireworks_client
from .config_schema import TrainingParams, LoRAParams

REGISTRY_PATH = Path(__file__).parent.parent / "registry.json"

def _update_lora_registry(lora_name: str, provider: str, params: TrainingParams):
    """Adds a new entry to the LoRA registry."""
    entry = {
        "name": lora_name,
        "provider": provider,
        "model_id": params.output_model,
        "base_model": params.base_model,
        "trained_at": datetime.utcnow().isoformat(),
    }
    
    registry = []
    if REGISTRY_PATH.exists():
        with REGISTRY_PATH.open("r") as f:
            registry = json.load(f)
            
    registry.append(entry)
    
    with REGISTRY_PATH.open("w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"Successfully added '{lora_name}' to the registry.")

def run_sft_job(
    lora_name: str,
    training_params: TrainingParams,
    lora_params: LoRAParams,
    client: fireworks_client,
):
    """
    Launches and monitors a Supervised Fine-Tuning (SFT) job on Fireworks.ai.
    """
    print("Launching SFT job...")
    job_id = client.launch_sft(training_params, lora_params)
    print(f"SFT job launched with ID: {job_id}")

    while True:
        status = client.get_sft_job_status(job_id)
        print(f"Job status: {status.get('status')}")
        if status.get("status") == "completed":
            print("Job completed successfully.")
            _update_lora_registry(lora_name, "fireworks", training_params)
            break
        elif status.get("status") in ["failed", "cancelled"]:
            print(f"Job failed or was cancelled. Status: {status.get('status')}")
            break
        time.sleep(60)