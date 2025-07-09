import os
from pathlib import Path
import sys
from src.lora.lora_manager import fine_tune

if __name__ == "__main__":
    dataset = Path(sys.argv[1])
    fine_tune(dataset, base_model=os.getenv("BASE_MODEL"))