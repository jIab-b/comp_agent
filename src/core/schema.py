from pydantic import BaseModel, Field
from typing import List, Optional

class agent_task(BaseModel):
    description: str
    context: Optional[str] = None

class agent_result(BaseModel):
    output: str
    iterations: int

class Example(BaseModel):
    messages: List[dict]
    meta: dict = Field(default_factory=dict)