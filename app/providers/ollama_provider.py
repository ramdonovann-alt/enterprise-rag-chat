"""
Unified Ollama provider for BOTH:
- Local Ollama:  http://192.168.1.8:11434/api/...
- Ollama Cloud: https://ollama.com/api/...

Both use the same native Ollama API paths:
- /api/chat
- /api/generate
- /api/embed

Switch by .env:
    OLLAMA_PROVIDER=local   # or cloud
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv

load_dotenv()


class OllamaProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaConfig:
    provider: str
    base_url: str
    api_key: str
    chat_model: str
    generate_model: str
    embed_model: str
    embed_base_url: str  # separate server for embeddings (optional)
    timeout: int
    retry: int
    num_ctx: Optional[int]


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int_env(name: str, default: int) -> int:
    value = _env(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _optional_int_env(name: str) -> Optional[int]:
    value = _env(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


class OllamaProvider:
    def __init__(self) -> None:
        provider = _env("OLLAMA_PROVIDER", "local").lower()
        if provider not in {"local", "cloud"}:
            raise OllamaProviderError("OLLAMA_PROVIDER must be 'local' or 'cloud'")

        if provider == "cloud":
            base_url = _env("OLLAMA_CLOUD_BASE_URL", "https://ollama.com")
            api_key = _env("OLLAMA_CLOUD_API_KEY", _env("OLLAMA_API_KEY", ""))
            chat_model = _env("OLLAMA_CLOUD_CHAT_MODEL", _env("OLLAMA_CHAT_MODEL", "gpt-oss:20b-cloud"))
            generate_model = _env("OLLAMA_CLOUD_GENERATE_MODEL", _env("OLLAMA_GENERATE_MODEL", chat_model))
            embed_model = _env("OLLAMA_CLOUD_EMBED_MODEL", _env("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
        else:
            base_url = _env("OLLAMA_LOCAL_BASE_URL", _env("OLLAMA_BASE_URL", "http://localhost:11434"))
            api_key = _env("OLLAMA_LOCAL_API_KEY", "")
            chat_model = _env("OLLAMA_LOCAL_CHAT_MODEL", _env("OLLAMA_CHAT_MODEL", "qwen2.5:14b-instruct"))
            generate_model = _env("OLLAMA_LOCAL_GENERATE_MODEL", _env("OLLAMA_GENERATE_MODEL", chat_model))
            embed_model = _env("OLLAMA_LOCAL_EMBED_MODEL", _env("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

        # OLLAMA_EMBED_BASE_URL overrides base_url for embedding requests only
        embed_base_url = _env("OLLAMA_EMBED_BASE_URL", base_url)

        self.config = OllamaConfig(
            provider=provider,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            chat_model=chat_model,
            generate_model=generate_model,
            embed_model=embed_model,
            embed_base_url=embed_base_url.rstrip("/"),
            timeout=_int_env("OLLAMA_TIMEOUT", 300),
            retry=_int_env("OLLAMA_RETRY", 3),
            num_ctx=_optional_int_env("OLLAMA_NUM_CTX"),
        )


    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def base_url(self) -> str:
        return self.config.base_url

    @property
    def chat_model(self) -> str:
        return self.config.chat_model

    @property
    def generate_model(self) -> str:
        return self.config.generate_model

    @property
    def embed_model(self) -> str:
        return self.config.embed_model

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        # Local Ollama usually does not need Authorization.
        # Cloud Ollama requires Bearer token.
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.retry + 1):
            try:
                response = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as ex:
                body = ""
                try:
                    body = response.text[:1000]
                except Exception:
                    pass
                last_error = OllamaProviderError(
                    f"HTTP {response.status_code} calling {url}: {body}"
                )
            except Exception as ex:
                last_error = ex

            if attempt < self.config.retry:
                time.sleep(min(2 * attempt, 5))

        raise OllamaProviderError(
            f"Request failed after {self.config.retry} retries: {last_error}"
        )

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.config.chat_model,
            "messages": list(messages),
            "stream": stream,
        }
        merged_options = dict(options or {})
        if self.config.num_ctx and "num_ctx" not in merged_options:
            merged_options["num_ctx"] = self.config.num_ctx
        if merged_options:
            payload["options"] = merged_options

        data = self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    def chat_stream(
        self,
        messages: Sequence[Dict[str, str]],
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Yield answer tokens from Ollama /api/chat with stream=True."""
        payload: Dict[str, Any] = {
            "model": model or self.config.chat_model,
            "messages": list(messages),
            "stream": True,
        }
        merged_options = dict(options or {})
        if self.config.num_ctx and "num_ctx" not in merged_options:
            merged_options["num_ctx"] = self.config.num_ctx
        if merged_options:
            payload["options"] = merged_options

        url = f"{self.config.base_url}/api/chat"
        with requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.config.timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                error = data.get("error")
                if error:
                    raise OllamaProviderError(str(error))
                token = (data.get("message") or {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.config.generate_model,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        merged_options = dict(options or {})
        if self.config.num_ctx and "num_ctx" not in merged_options:
            merged_options["num_ctx"] = self.config.num_ctx
        if merged_options:
            payload["options"] = merged_options

        data = self._post("/api/generate", payload)
        return data.get("response", "")

    def _post_embed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to embed_base_url — no auth header for local servers."""
        url = f"{self.config.embed_base_url}/api/embed"
        # Only send auth header if the embed server is the same as the main server
        headers = {"Content-Type": "application/json"}
        if self.config.embed_base_url == self.config.base_url and self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.retry + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as ex:
                body = ""
                try:
                    body = response.text[:1000]
                except Exception:
                    pass
                last_error = OllamaProviderError(f"HTTP {response.status_code} calling {url}: {body}")
            except Exception as ex:
                last_error = ex
            if attempt < self.config.retry:
                time.sleep(min(2 * attempt, 5))
        raise OllamaProviderError(f"Embed request failed after {self.config.retry} retries: {last_error}")

    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        payload = {
            "model": model or self.config.embed_model,
            "input": text,
        }
        data = self._post_embed(payload)

        # Native Ollama /api/embed returns {"embeddings": [[...]]}
        if "embeddings" in data and data["embeddings"]:
            return data["embeddings"][0]

        # Compatibility fallback for older /api/embeddings style payloads.
        if "embedding" in data:
            return data["embedding"]

        raise OllamaProviderError(f"Embedding response has no embedding data: {data}")

    def embeddings(self, texts: Sequence[str], model: Optional[str] = None) -> List[List[float]]:
        payload = {
            "model": model or self.config.embed_model,
            "input": list(texts),
        }
        data = self._post_embed(payload)
        if "embeddings" in data:
            return data["embeddings"]
        raise OllamaProviderError(f"Embedding response has no embeddings data: {data}")

    def health(self) -> Dict[str, Any]:
        # /api/tags is a GET endpoint in Ollama. Use requests directly.
        url = f"{self.config.base_url}/api/tags"
        response = requests.get(url, headers=self._headers(), timeout=30)
        response.raise_for_status()
        return {
            "provider": self.config.provider,
            "base_url": self.config.base_url,
            "chat_model": self.config.chat_model,
            "generate_model": self.config.generate_model,
            "embed_model": self.config.embed_model,
            "tags": response.json(),
        }


_default_provider: Optional[OllamaProvider] = None


def get_ollama_provider() -> OllamaProvider:
    global _default_provider
    if _default_provider is None:
        _default_provider = OllamaProvider()
    return _default_provider


def chat_text(prompt: str, system: Optional[str] = None) -> str:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return get_ollama_provider().chat(messages)


def generate_text(prompt: str, system: Optional[str] = None) -> str:
    return get_ollama_provider().generate(prompt=prompt, system=system)


def embed_text(text: str) -> List[float]:
    return get_ollama_provider().embed(text)
