"""LLM client wrapper supporting OpenAI-compatible APIs.

Provides both free-text completion and structured (JSON mode) output,
with retry logic and token tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMClient:
    """Async OpenAI-compatible LLM client.

    Supports standard OpenAI and any OpenAI-compatible endpoint.
    Set OPENAI_BASE_URL for custom endpoints.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.max_retries = max_retries

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL")

        self._client = AsyncOpenAI(
            api_key=resolved_key or None,
            base_url=resolved_base,
        )
        self._total_tokens = 0

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Free-text completion."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response.usage:
                    self._total_tokens += response.usage.total_tokens
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(min(2 ** attempt, 16))
        return ""

    async def complete_structured(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """JSON-mode completion — returns parsed dict."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                if response.usage:
                    self._total_tokens += response.usage.total_tokens
                text = response.choices[0].message.content or "{}"
                return json.loads(text)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return {}
                await asyncio.sleep(min(2 ** attempt, 16))
            except Exception as e:
                logger.warning(f"LLM structured attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(min(2 ** attempt, 16))
        return {}

    async def complete_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """Chat completion with full message history."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                if response.usage:
                    self._total_tokens += response.usage.total_tokens
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"Chat attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(min(2 ** attempt, 16))
        return ""
