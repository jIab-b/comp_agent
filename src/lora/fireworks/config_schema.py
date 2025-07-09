from pydantic import BaseModel, Field
from typing import List, Optional, Union

class LoRAParams(BaseModel):
    r: int = Field(8, ge=4, le=64, description="LoRA rank")
    alpha: int = 8
    dropout: float = 0.0
    target_modules: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "up_proj", "down_proj", "gate_proj"
    ]

class TrainingParams(BaseModel):
    base_model: str
    dataset_id: str
    output_model: str
    learning_rate: float = 1e-4
    epochs: int = 1
    batch_size: Union[str, int] = "max"
    early_stop: bool = False
    max_context_length: Optional[int] = None
    turbo: bool = False