"""Unified LLM client supporting both Claude (Anthropic) and OpenAI APIs."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai

logger = logging.getLogger(__name__)

from autospace.llm.logger import LLMLogger


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_s: float
    raw_response: Any = field(repr=False, default=None)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient:
    """Unified client for Claude and OpenAI models.

    Routes requests based on model name prefix:
      - claude-* → Anthropic SDK
      - gpt-*    → OpenAI SDK
    """

    def __init__(
        self,
        model: str,
        anthropic_api_key: str = "",
        openai_api_key: str = "",
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = 16384,
        logger: LLMLogger | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.logger = logger

        # Running token totals
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Initialize appropriate client
        if self.is_claude:
            self._anthropic = anthropic.Anthropic(api_key=anthropic_api_key)
        elif self.is_openai:
            self._openai = openai.OpenAI(api_key=openai_api_key)
        else:
            raise ValueError(
                f"Unknown model provider for '{model}'. "
                "Model must start with 'claude-' or 'gpt-'."
            )

    @property
    def is_claude(self) -> bool:
        return self.model.startswith("claude")

    @property
    def is_openai(self) -> bool:
        return self.model.startswith("gpt")

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat request and return a standardized response.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts.
            system: System prompt (handled differently per provider).
            temperature: Override default temperature for this call.
            max_tokens: Override default max_tokens for this call.

        Returns:
            LLMResponse with content, token counts, and latency.
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        start = time.monotonic()
        response = self._call_with_retry(messages, system, temp, max_tok)
        response.latency_s = time.monotonic() - start

        # Track totals
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens

        # Log if logger is configured
        if self.logger:
            self.logger.log_interaction(
                model=self.model,
                system=system,
                messages=messages,
                response_content=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_s=response.latency_s,
            )

        return response

    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        max_tokens: int,
        max_retries: int = 5,
        base_delay: float = 2.0,
    ) -> LLMResponse:
        """Call LLM with exponential backoff retry for transient errors."""
        for attempt in range(1, max_retries + 1):
            try:
                if self.is_claude:
                    return self._call_anthropic(messages, system, temperature, max_tokens)
                else:
                    return self._call_openai(messages, system, temperature, max_tokens)
            except (
                anthropic.RateLimitError,
                anthropic.InternalServerError,
                anthropic.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.APIConnectionError,
                ConnectionError,
                TimeoutError,
            ) as e:
                if attempt == max_retries:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"API error (attempt {attempt}/{max_retries}): {e}. "
                    f"Retrying in {delay:.0f}s..."
                )
                time.sleep(delay)
            except openai.BadRequestError as e:
                # Retry JSON body parse errors (transient SDK/network issue),
                # but propagate other 400 errors (e.g., invalid model) immediately.
                if "parse the JSON body" not in str(e):
                    raise
                if attempt == max_retries:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"OpenAI JSON body error (attempt {attempt}/{max_retries}): {e}. "
                    f"Retrying in {delay:.0f}s..."
                )
                time.sleep(delay)
        raise RuntimeError("Unreachable")

    _STREAM_THRESHOLD: int = 16384

    def _call_anthropic(
        self,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        if max_tokens > self._STREAM_THRESHOLD:
            content = ""
            with self._anthropic.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    content += text
                final = stream.get_final_message()
            return LLMResponse(
                content=content,
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                model=final.model,
                latency_s=0.0,
                raw_response=final,
            )

        resp = self._anthropic.messages.create(**kwargs)

        content = ""
        for block in resp.content:
            if block.type == "text":
                content += block.text

        return LLMResponse(
            content=content,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=resp.model,
            latency_s=0.0,
            raw_response=resp,
        )

    @staticmethod
    def _sanitize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Ensure all message content values are clean strings (no NaN/Inf floats)."""
        sanitized = []
        for msg in messages:
            sanitized.append({
                k: str(v) if not isinstance(v, str) else v
                for k, v in msg.items()
            })
        return sanitized

    def _call_openai(
        self,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)
        oai_messages = self._sanitize_messages(oai_messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,  # GPT-5.4+ requires this instead of max_tokens
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed

        if max_tokens > self._STREAM_THRESHOLD:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            content = ""
            input_tokens = 0
            output_tokens = 0
            model_id = self.model
            for chunk in self._openai.chat.completions.create(**kwargs):
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content
                if getattr(chunk, "usage", None):
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
                if getattr(chunk, "model", None):
                    model_id = chunk.model
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model_id,
                latency_s=0.0,
                raw_response=None,  # streaming yields chunks, not a single response object
            )

        resp = self._openai.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            model=resp.model or self.model,
            latency_s=0.0,
            raw_response=resp,
        )

    @classmethod
    def from_config(cls, config, logger: LLMLogger | None = None) -> "LLMClient":
        """Create client from LLMConfig."""
        return cls(
            model=config.model,
            anthropic_api_key=config.anthropic_api_key,
            openai_api_key=config.openai_api_key,
            temperature=config.temperature,
            seed=config.seed,
            max_tokens=config.max_tokens,
            logger=logger,
        )
