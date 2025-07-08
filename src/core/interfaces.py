from abc import ABC, abstractmethod
from typing import List, Dict, Any

class llm(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        ...

class agent(ABC):
    @abstractmethod
    def run(self, task: str) -> str:
        ...

class memory(ABC):
    @abstractmethod
    def add(self, text: str):
        ...
    @abstractmethod
    def recall(self, query: str) -> List[str]:
        ...