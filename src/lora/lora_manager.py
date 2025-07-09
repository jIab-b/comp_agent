import subprocess
import yaml
import pathlib
import uuid

def fine_tune(dataset_path: str, base_model: str) -> str:
    output_model = f"myagent-lora-{uuid.uuid4().hex[:8]}"
    cmd = [
        "firectl", "create", "sftj",
        "--base-model", base_model,
        "--dataset", dataset_path,
        "--output-model", output_model,
    ]
    subprocess.check_call(cmd)
    _write_model_to_config(output_model)
    return output_model

def _write_model_to_config(model_id: str):
    cfg_path = pathlib.Path(__file__).parents[2] / "config" / "settings.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["llm"]["fireworks"]["lora_model"] = model_id
    cfg_path.write_text(yaml.safe_dump(cfg))