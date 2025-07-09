from pathlib import Path
from .builder import validate_examples, write_jsonl
from .builder.converters import convert_file
from .fireworks.client import fireworks_client
from .fireworks.config_schema import TrainingParams, LoRAParams
from .fireworks.sft_job import run_sft_job
from .fireworks.upload import upload_dataset

def build_dataset(dataset_dir: Path) -> Path:
    """
    Processes raw files from a specific dataset directory, validates them,
    and packs them into a JSONL file in the 'processed' directory.
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    all_examples = []
    for raw_file in dataset_dir.iterdir():
        if raw_file.is_file():
            all_examples.extend(convert_file(raw_file))
    
    validate_examples(all_examples)
    
    processed_dir = Path(__file__).parent / "data/processed"
    processed_dir.mkdir(exist_ok=True)
    
    output_path = processed_dir / f"{dataset_dir.name}.jsonl"
    write_jsonl(all_examples, output_path)
    
    return output_path

def train_lora(
    lora_name: str,
    dataset_name: str,
    cfg: TrainingParams,
    lora_cfg: LoRAParams,
):
    """
    Orchestrates the entire LoRA training pipeline.
    """
    # 1. Build dataset
    raw_data_dir = Path(__file__).parent / "data/raw"
    dataset_path = build_dataset(raw_data_dir / dataset_name)
    
    # 2. Initialize client
    fw_client = fireworks_client()
    
    # 3. Upload dataset
    upload_dataset(dataset_path, cfg.dataset_id, fw_client)
    
    # 4. Launch and monitor SFT job
    run_sft_job(lora_name, cfg, lora_cfg, fw_client)