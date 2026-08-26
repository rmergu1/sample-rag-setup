"""
Routes a chat completion request to one of three free-tier LLM providers:
groq, google (gemini), openrouter -- selected by request body, defaulting to
DEFAULT_PROVIDER (groq) from .env.
"""
import os
from typing import Optional, Tuple

import httpx

PROVIDERS = ["groq", "google", "openrouter"]


class ProviderError(Exception):
    pass


def resolve_provider_and_model(provider: Optional[str], model: Optional[str]) -> Tuple[str, str]:
    """
    Figures out which provider + model to use for a request.

    Accepted request shapes:
      {}                                              -> DEFAULT_PROVIDER + its default model
      {"model": "llama-3.1-8b-instant"}                -> DEFAULT_PROVIDER + given model
      {"model": "google/gemini-1.5-flash"}             -> provider inferred from "google/" prefix
      {"provider": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:free"}
                                                        -> explicit provider, model used as-is
                                                           (openrouter model names contain slashes,
                                                           so prefix-splitting is skipped when
                                                           provider is given explicitly)
    """
    default_provider = os.getenv("DEFAULT_PROVIDER", "groq")

    if provider is None and model and "/" in model:
        prefix, rest = model.split("/", 1)
        if prefix in PROVIDERS:
            provider = prefix
            model = rest

    provider = (provider or default_provider).lower()
    if provider not in PROVIDERS:
        raise ProviderError(f"Unknown provider '{provider}'. Must be one of {PROVIDERS}")

    if not model:
        model = os.getenv(f"{provider.upper()}_DEFAULT_MODEL")
        if not model:
            raise ProviderError(f"No model given and {provider.upper()}_DEFAULT_MODEL not set in .env")

    return provider, model


async def call_llm(provider: str, model: str, system_prompt: str, user_prompt: str) -> str:
    if provider in ("groq", "openrouter"):
        return await _call_openai_compatible(provider, model, system_prompt, user_prompt)
    if provider == "google":
        return await _call_google(model, system_prompt, user_prompt)
    raise ProviderError(f"Unsupported provider: {provider}")


async def _call_openai_compatible(provider: str, model: str, system_prompt: str, user_prompt: str) -> str:
    base_url = os.getenv(f"{provider.upper()}_BASE_URL")
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    if not base_url:
        raise ProviderError(f"{provider.upper()}_BASE_URL is not set in .env")
    if not api_key:
        raise ProviderError(f"{provider.upper()}_API_KEY is not set in .env")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        #"temperature": 0.2,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)

    if resp.status_code != 200:
        raise ProviderError(f"{provider} error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise ProviderError(f"Unexpected {provider} response shape: {data}")


async def _call_google(model: str, system_prompt: str, user_prompt: str) -> str:
    base_url = os.getenv("GOOGLE_BASE_URL")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not base_url:
        raise ProviderError("GOOGLE_BASE_URL is not set in .env")
    if not api_key:
        raise ProviderError("GOOGLE_API_KEY is not set in .env")

    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    payload = {"contents": [{"parts": [{"text": combined_prompt}]}]}
    url = f"{base_url}/models/{model}:generateContent?key={api_key}"

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)

    if resp.status_code != 200:
        raise ProviderError(f"google error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise ProviderError(f"Unexpected google response shape: {data}")
