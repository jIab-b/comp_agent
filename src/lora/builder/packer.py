import json
from pathlib import Path
from typing import List, Optional
from src.core.schema import Example

def write_jsonl(
    examples: List[Example],
    output_path: Path,
    shard_size: Optional[int] = None
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if shard_size:
        _write_sharded_jsonl(examples, output_path, shard_size)
    else:
        _write_single_jsonl(examples, output_path)

def _write_single_jsonl(examples: List[Example], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            # Using asdict() if Example is a dataclass
            f.write(json.dumps(example.dict()) + "\n")

def _write_sharded_jsonl(
    examples: List[Example],
    output_path: Path,
    shard_size: int
):
    for i in range(0, len(examples), shard_size):
        shard_examples = examples[i : i + shard_size]
        shard_path = output_path.with_name(
            f"{output_path.stem}_{i // shard_size}{output_path.suffix}"
        )
        _write_single_jsonl(shard_examples, shard_path)