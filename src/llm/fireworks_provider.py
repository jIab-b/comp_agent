import os
from langchain_community.chat_models import ChatOpenAI
from src.core.interfaces import llm

class fireworks_chat(ChatOpenAI, llm):
    def __init__(self,
                 model: str,
                 temperature: float = 0.2,
                 streaming: bool = False,
                 **openai_kwargs):
        super().__init__(
            openai_api_key=os.getenv("FIREWORKS_API_KEY"),
            openai_api_base="https://api.fireworks.ai/v1",
            model=model,
            temperature=temperature,
            streaming=streaming,
            **openai_kwargs,
        )

    def chat(self, messages: list, **kwargs) -> str:
        response = self.invoke(messages, **kwargs)
        return response.content