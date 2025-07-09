import subprocess
from pathlib import Path
from .config_schema import TrainingParams, LoRAParams

class fireworks_client:
    def _run_command(self, cmd: list[str]):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {' '.join(cmd)}")
            print(f"Stderr: {e.stderr}")
            raise

    def create_dataset(self, dataset_path: Path, dataset_id: str):
        cmd = ["firectl", "create", "dataset", dataset_id, str(dataset_path)]
        self._run_command(cmd)

    def launch_sft(self, p: TrainingParams, lora: LoRAParams):
        cmd = [
            "firectl", "create", "sftj",
            "--base-model", p.base_model,
            "--dataset", p.dataset_id,
            "--output-model", p.output_model,
            "--learning-rate", str(p.learning_rate),
            "--epochs", str(p.epochs),
            "--batch-size", str(p.batch_size),
            "--lora-r", str(lora.r),
            "--lora-alpha", str(lora.alpha),
            "--lora-dropout", str(lora.dropout),
            "--lora-trainable-modules", ",".join(lora.target_modules),
        ]
        if p.early_stop:
            cmd.append("--early-stop")
        if p.max_context_length:
            cmd += ["--max-context-length", str(p.max_context_length)]
        if p.turbo:
            cmd.append("--turbo")
        
        self._run_command(cmd)

    def get_sft_job_status(self, job_id: str):
        cmd = ["firectl", "get", "sftj", job_id]
        result = self._run_command(cmd)
        # This assumes the output is JSON, which may need adjustment
        # based on the actual output of `firectl`.
        import json
        return json.loads(result.stdout)