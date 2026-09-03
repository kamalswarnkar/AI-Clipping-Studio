"""Ollama-backed LLM and vision providers -- fully local inference.

Two things make local models usable for this pipeline:

1. Ollama's ``format`` parameter accepts a JSON Schema and constrains decoding to
   it, so the model *cannot* emit malformed JSON.
2. Everything is validated again on our side anyway. Schema-constrained decoding
   still allows semantically wrong values, and nothing downstream acts on model
   output that has not been re-checked.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import httpx

from ...config import get_settings
from ..base import ProviderError, ProviderUnavailable

log = logging.getLogger(__name__)

# Models sometimes wrap JSON in prose or fences despite constraints; salvage it.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except ValueError:
        pass

    fenced = _FENCE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except ValueError:
            pass

    # Fall back to the outermost balanced object/array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                continue

    raise ValueError(f"no JSON found in response: {text[:200]}")


class _OllamaBase:
    """Shared HTTP plumbing for the chat endpoint."""

    def __init__(self, model: str) -> None:
        s = get_settings()
        self.base_url = s.ollama_base_url.rstrip("/")
        self.model = model
        self.timeout = s.ollama_timeout
        self.num_ctx = s.ollama_num_ctx
        self.default_temperature = s.ollama_temperature
        self.keep_alive = s.ollama_keep_alive
        self.json_retries = max(1, s.ollama_json_retries)

    # --- availability -------------------------------------------------------
    def _list_models(self) -> list[str]:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]

    def is_available(self) -> tuple[bool, str]:
        try:
            names = self._list_models()
        except Exception as exc:  # noqa: BLE001
            return False, (
                f"Cannot reach Ollama at {self.base_url} ({exc}). "
                "Start it with: ollama serve"
            )

        if not names:
            return False, (
                f"Ollama is running but has no models. Run: ollama pull {self.model}"
            )

        # Ollama reports 'name:tag'; accept an exact match or a bare-name match.
        bare = self.model.split(":")[0]
        if self.model in names or any(n.split(":")[0] == bare for n in names):
            return True, ""

        return False, (
            f"Model '{self.model}' is not installed in Ollama "
            f"(found: {', '.join(names[:5])}). Run: ollama pull {self.model}"
        )

    # --- request ------------------------------------------------------------
    def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        schema: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        options: dict[str, Any] = {
            "temperature": self.default_temperature if temperature is None else temperature,
            "num_ctx": self.num_ctx,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        if schema is not None:
            body["format"] = schema

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/api/chat", json=body)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"The local model timed out after {self.timeout:.0f}s. "
                "Try a smaller model or raise OLLAMA_TIMEOUT.",
                provider="ollama",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"Cannot reach Ollama at {self.base_url}: {exc}",
                provider="ollama",
            ) from exc

        if resp.status_code == 404:
            raise ProviderUnavailable(
                f"Ollama does not have the model '{self.model}'. "
                f"Run: ollama pull {self.model}",
                provider="ollama",
            )
        if resp.status_code >= 400:
            raise ProviderError(
                f"Ollama returned {resp.status_code}: {resp.text[:300]}",
                provider="ollama",
                retryable=resp.status_code >= 500,
            )

        try:
            return resp.json()["message"]["content"]
        except (ValueError, KeyError) as exc:
            raise ProviderError(
                "Unexpected response shape from Ollama.", provider="ollama"
            ) from exc


class OllamaLLMProvider(_OllamaBase):
    name = "ollama"

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__(model or get_settings().ollama_llm_model)

    def complete_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Get schema-constrained JSON, retrying on unparseable output."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        last_error: Exception | None = None

        for attempt in range(self.json_retries):
            raw = self._chat(
                messages,
                schema=schema,
                # Nudge toward determinism on retries.
                temperature=0.0 if attempt else temperature,
                max_tokens=max_tokens,
            )
            try:
                parsed = _extract_json(raw)
            except ValueError as exc:
                last_error = exc
                log.warning(
                    "Ollama returned unparseable JSON (attempt %d/%d): %s",
                    attempt + 1,
                    self.json_retries,
                    exc,
                )
                messages.append({"role": "assistant", "content": raw[:500]})
                messages.append(
                    {
                        "role": "user",
                        "content": "That was not valid JSON. Reply with the JSON object only.",
                    }
                )
                continue

            if isinstance(parsed, list):
                parsed = {"items": parsed}
            if not isinstance(parsed, dict):
                last_error = ValueError(f"expected object, got {type(parsed).__name__}")
                continue
            return parsed

        raise ProviderError(
            f"The local model did not return valid JSON after {self.json_retries} "
            f"attempts ({last_error}).",
            provider="ollama",
            retryable=True,
        )

    def complete_text(
        self,
        *,
        system: str,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        ).strip()


class OllamaVisionProvider(_OllamaBase):
    """Multimodal reasoning over sampled frames.

    Deliberately bounded: only the strongest candidates get a call, and each call
    sends a handful of frames rather than a video stream.
    """

    name = "ollama"

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__(model or get_settings().ollama_vision_model)

    def describe_frames(
        self,
        image_paths: list[Path],
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        images: list[str] = []
        for p in image_paths:
            if p.exists():
                images.append(base64.b64encode(p.read_bytes()).decode("ascii"))
        if not images:
            raise ProviderError("No frames available to analyse.", provider="ollama")

        raw = self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt, "images": images},
            ],
            schema=schema,
            temperature=0.1,
        )
        try:
            parsed = _extract_json(raw)
        except ValueError as exc:
            raise ProviderError(
                f"Vision model returned unparseable JSON: {exc}",
                provider="ollama",
                retryable=True,
            ) from exc

        return parsed if isinstance(parsed, dict) else {"items": parsed}


class NullLLMProvider:
    """Used when LLM_PROVIDER=null. Fails loudly rather than silently degrading."""

    name = "null"
    model = "none"

    def is_available(self) -> tuple[bool, str]:
        return False, "LLM is disabled (LLM_PROVIDER=null)."

    def complete_json(self, **_: Any) -> dict[str, Any]:
        raise ProviderUnavailable("LLM is disabled.", provider="null")

    def complete_text(self, **_: Any) -> str:
        raise ProviderUnavailable("LLM is disabled.", provider="null")


class NullVisionProvider:
    name = "null"
    model = "none"

    def is_available(self) -> tuple[bool, str]:
        return False, "Vision analysis is disabled (VISION_PROVIDER=null)."

    def describe_frames(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise ProviderUnavailable("Vision is disabled.", provider="null")
