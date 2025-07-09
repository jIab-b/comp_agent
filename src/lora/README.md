# LoRA Fine-Tuning Module

This module provides a complete pipeline for fine-tuning models using LoRA with Fireworks.ai. It includes tools for data preparation, validation, training, and a registry for managing your fine-tuned models.

## Directory Structure

- `data/`: Staging area for datasets.
  - `raw/`: Place your raw data files here (e.g., `.json`, `.csv`, `.md`).
  - `processed/`: Output directory for processed `.jsonl` files.
- `builder/`: Scripts for preparing raw data.
- `fireworks/`: Fireworks.ai API client and configuration.
- `orchestrator.py`: High-level functions for the training pipeline.
- `cli.py`: Command-line interface for the module.
- `registry.json`: A catalog of all your trained LoRA models.

## Usage Workflow

### 1. Stage Your Files
Use the `stage` command to copy your source files into a dedicated dataset directory. This command creates the directory if it doesn't exist and leaves your original files untouched.

For example, to stage two PDF files for a dataset named `policy-docs`:
```bash
python -m src.lora.cli stage policy-docs ~/Documents/handbook.pdf /data/archive/code_of_conduct.pdf
```
This creates the `src/lora/data/raw/policy-docs/` directory and copies the specified files into it.

### 2. Build the Dataset
Use the `build` command, specifying the name of the dataset directory you want to process.
```bash
python -m src.lora.cli build policy-docs
```
This command reads all files from `src/lora/data/raw/policy-docs/`, validates them, and creates a single training-ready file at `src/lora/data/processed/policy-docs.jsonl`.

### 3. Launch a Training Job
Use the `train` command to launch a fine-tuning job. Provide a unique `--name` for the registry and specify the `--dataset_name` to use.
```bash
python -m src.lora.cli train \
  --name "policy-expert-v1" \
  --dataset_name "policy-docs" \
  --base-model "accounts/fireworks/models/llama-v3p1-8b-instruct" \
  --output-model "my-org/llama3-policy-expert-v1" \
  --epochs 3
```
Upon completion, the new model will be automatically added to `registry.json`.

### 4. List Available LoRAs
To see all your trained models, use the `list` command.
```bash
python -m src.lora.cli list
```

### 5. Use Your Fine-Tuned Model
To use a specific LoRA model for inference, use the `--lora` flag with its registry name when running the main agent.
```bash
python src/cli/run_agent.py --lora "policy-expert-v1"
```
The application will automatically load the correct model from the registry. If the `--lora` flag is omitted, the default model from `config/settings.yaml` will be used.

## Command-Line Interface

#### `stage`
- `dataset_name`: The name of the dataset directory to create/use.
- `source_files`: One or more paths to the source files to copy.

#### `build`
- `dataset_name`: The name of the dataset directory inside `src/lora/data/raw/`.

#### `list`
- Lists all registered LoRA models.

#### `train`
- `--name`: A unique name for the LoRA model in the registry.
- `--dataset_name`: The name of the dataset to use for training.
- `--base_model`: The base model to fine-tune.
- `--output_model`: The name for your fine-tuned model on Fireworks.ai.
- ... and other training/LoRA parameters.