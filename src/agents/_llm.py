"""Shared Azure OpenAI client with an opt-in local Ollama fallback."""
import json
import logging
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from openai import APIError, AzureOpenAI, BadRequestError
from tiktoken import get_encoding

load_dotenv()

_client = None
MAX_INPUT_TOKENS = 96_000


def count_context_tokens(system: str, user: str) -> int:
    messages = system + "\n" + user
    return max(len(get_encoding(name).encode(messages, disallowed_special=()))
               for name in ("cl100k_base", "o200k_base")) + 16


def validate_context(system: str, user: str, max_tokens: int = MAX_INPUT_TOKENS) -> None:
    tokens = count_context_tokens(system, user)
    if tokens > max_tokens:
        raise ValueError(
            f"Documents need approximately {tokens:,} input tokens; the limit is {max_tokens:,}. "
            "Upload fewer worksheets or only the relevant constraints rows. No document text was sent to the model."
        )


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        )
    return _client


def _ollama_call(system: str, user: str, temperature: float, json_mode: bool) -> str:
    context_tokens = int(os.environ.get("OLLAMA_CONTEXT_TOKENS", "8192"))
    validate_context(system, user, max_tokens=max(0, context_tokens - 2048))
    default_host = "host.docker.internal" if Path("/.dockerenv").exists() else "127.0.0.1"
    base_url = os.environ.get("OLLAMA_BASE_URL", f"http://{default_host}:11434").rstrip("/")
    payload = {
        "model": os.environ.get("OLLAMA_MODEL", "hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M"),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": context_tokens, "num_predict": 2048},
    }
    if json_mode:
        payload["format"] = "json"
    request = Request(f"{base_url}/api/chat", data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        content = result.get("message", {}).get("content")
        if result.get("error") or not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama returned no usable response.")
        if result.get("done_reason") == "length":
            raise ValueError("Ollama response exceeded its output limit.")
        return content
    except (URLError, OSError, ValueError) as error:
        raise RuntimeError("Azure failed and the Ollama fallback could not complete. "
                           "Check that Ollama is running and the fallback model is pulled.") from error


def call(system: str, user: str, *, model: str | None = None, temperature: float = 0.0,
         json_mode: bool = False) -> str:
    """Call the LLM and return the raw text response."""
    validate_context(system, user)
    load_dotenv(override=True)   # re-read .env on every process start (handles stale module cache)
    deployment = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    try:
        resp = _get_client().chat.completions.create(
            model=deployment,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
    except (APIError, KeyError) as error:
        if isinstance(error, BadRequestError):
            details = error.body if isinstance(error.body, dict) else {}
            details = details.get("error", details)
            if isinstance(details, dict) and (
                details.get("code") == "context_length_exceeded"
                or "maximum context length" in str(details.get("message", "")).lower()
            ):
                raise ValueError(
                    "The model rejected this request because it exceeds its context window. "
                    "Upload fewer worksheets or only the relevant constraints rows. "
                    "The request was not retried with the smaller local model, and no extraction was saved."
                ) from None
        enabled = os.environ.get("OLLAMA_FALLBACK_ENABLED", "false").lower() in ("true", "1", "yes")
        if not enabled or isinstance(error, BadRequestError):
            raise
        logging.getLogger(__name__).warning("Azure unavailable (%s); using local Ollama fallback.", type(error).__name__)
        return _ollama_call(system, user, temperature, json_mode)
    if resp.choices[0].finish_reason == "length":
        raise ValueError("The model response was truncated. No complete extraction was produced; reduce the document batch size.")
    return resp.choices[0].message.content


def call_json(system: str, user: str, **kwargs) -> dict | list:
    """Call the LLM and parse the response as JSON (strips markdown fences if present)."""
    raw = call(system, user, json_mode=True, **kwargs)
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)
