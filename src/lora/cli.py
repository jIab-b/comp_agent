import argparse
import json
import shutil
from pathlib import Path
from .orchestrator import build_dataset, train_lora
from .fireworks.config_schema import TrainingParams, LoRAParams

REGISTRY_PATH = Path(__file__).parent / "registry.json"
RAW_DATA_DIR = Path(__file__).parent / "data/raw"

def list_loras():
    """Prints a formatted table of all registered LoRA models."""
    if not REGISTRY_PATH.exists():
        print("No LoRA models found in the registry.")
        return
        
    with REGISTRY_PATH.open("r") as f:
        registry = json.load(f)
        
    if not registry:
        print("No LoRA models found in the registry.")
        return
        
    print(f"{'Name':<20} {'Provider':<15} {'Base Model':<40} {'Trained At':<25}")
    print("-" * 100)
    for entry in registry:
        print(f"{entry['name']:<20} {entry['provider']:<15} {entry['base_model']:<40} {entry['trained_at']:<25}")

def stage_files(dataset_name: str, source_files: list[Path]):
    """Copies source files to the specified dataset directory."""
    dataset_dir = RAW_DATA_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    for source_file in source_files:
        if not source_file.exists():
            print(f"Warning: Source file not found, skipping: {source_file}")
            continue
        destination = dataset_dir / source_file.name
        shutil.copy(source_file, destination)
        print(f"Copied '{source_file}' to '{destination}'")

def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stage command
    stage_parser = subparsers.add_parser("stage", help="Copy source files to a dataset directory.")
    stage_parser.add_argument("dataset_name", type=str, help="Name of the dataset directory.")
    stage_parser.add_argument("source_files", type=Path, nargs='+', help="One or more source files to copy.")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build a dataset from a directory of raw files.")
    build_parser.add_argument("dataset_name", type=str, help="Name of the dataset directory in src/lora/data/raw.")

    # List command
    list_parser = subparsers.add_parser("list", help="List all registered LoRA models.")

    # Train command
    train_parser = subparsers.add_parser("train", help="Launch a LoRA training job.")
    train_parser.add_argument("--name", type=str, required=True, help="A unique name for the LoRA model in the registry.")
    train_parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset directory to use.")
    # TrainingParams
    train_parser.add_argument("--base_model", type=str, required=True)
    train_parser.add_argument("--output_model", type=str, required=True)
    train_parser.add_argument("--learning_rate", type=float, default=1e-4)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch_size", type=str, default="max")
    train_parser.add_argument("--early_stop", action="store_true")
    train_parser.add_argument("--max_context_length", type=int)
    train_parser.add_argument("--turbo", action="store_true")
    # LoRAParams
    train_parser.add_argument("--r", type=int, default=8)
    train_parser.add_argument("--alpha", type=int, default=8)
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])

    args = parser.parse_args()

    if args.command == "stage":
        stage_files(args.dataset_name, args.source_files)
    elif args.command == "build":
        dataset_dir = RAW_DATA_DIR / args.dataset_name
        build_dataset(dataset_dir)
        print(f"Dataset '{args.dataset_name}' built successfully.")
    elif args.command == "list":
        list_loras()
    elif args.command == "train":
        training_params = TrainingParams(
            base_model=args.base_model,
            dataset_id=args.dataset_name,
            output_model=args.output_model,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stop=args.early_stop,
            max_context_length=args.max_context_length,
            turbo=args.turbo,
        )
        lora_params = LoRAParams(
            r=args.r,
            alpha=args.alpha,
            dropout=args.dropout,
            target_modules=args.target_modules,
        )
        train_lora(args.name, args.dataset_name, training_params, lora_params)

if __name__ == "__main__":
    main()