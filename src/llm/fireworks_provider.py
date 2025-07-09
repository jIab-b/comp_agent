import os
import httpx
import json
from typing import Any, List, Iterator, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk

from src.core.interfaces import llm

def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary."""
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    else:
        raise TypeError(f"Unsupported message type: {type(message)}")

class fireworks_chat(BaseChatModel, llm):
    """
    A custom chat model provider for Fireworks AI that inherits from BaseChatModel.
    """
    model: str
    temperature: float = 0.2
    streaming: bool = False
    api_base: str = "https://api.fireworks.ai/inference/v1"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a chat response from the Fireworks AI API.
        """
        if self.streaming:
            completion = ""
            for chunk in self._stream(messages, stop, run_manager, **kwargs):
                completion += chunk.message.content
            message = AIMessage(content=completion)
            return ChatResult(generations=[ChatGeneration(message=message)])

        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("Fireworks API key not set. Please set the FIREWORKS_API_KEY environment variable.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        message_dicts = [convert_message_to_dict(m) for m in messages]

        body = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
            "stop": stop,
            **kwargs,
        }

        with httpx.Client() as client:
            response = client.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=body,
                timeout=60,
            )
            response.raise_for_status()

        response_json = response.json()
        message = AIMessage(content=response_json["choices"][0]["message"]["content"])
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("Fireworks API key not set. Please set the FIREWORKS_API_KEY environment variable.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        message_dicts = [convert_message_to_dict(m) for m in messages]

        body = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
            "stop": stop,
            "stream": True,
            **kwargs,
        }

        with httpx.stream("POST", f"{self.api_base}/chat/completions", headers=headers, json=body, timeout=60) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(line)
                        delta = chunk_data["choices"][0]["delta"]
                        if "content" in delta and delta["content"] is not None:
                            message = AIMessageChunk(content=delta["content"])
                            chunk = ChatGenerationChunk(message=message)
                            yield chunk
                            if run_manager:
                                run_manager.on_llm_new_token(chunk.message.content)
                    except json.JSONDecodeError:
                        # Skip empty or malformed lines
                        pass

    @property
    def _llm_type(self) -> str:
        return "fireworks_chat"

    def chat(self, messages: list, **kwargs) -> str:
        response = self.invoke(messages, **kwargs)
        return response.content