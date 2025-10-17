from concurrent.futures import ThreadPoolExecutor
import os
import openai
from typing import Optional
import time
import asyncio

# Optional compatibility with newer OpenAI SDK (v1+)
try:
    from openai import OpenAI as _OpenAIClient  # type: ignore
except Exception:  # pragma: no cover
    _OpenAIClient = None  # type: ignore

def prompt_chatgpt(prompt: str = "Say hello.", *, model: str = "gpt-3.5-turbo", temperature: float = 0.0,
                    max_tokens: int = 256, timeout: Optional[float] = None) -> str:
    """Send a simple prompt to ChatGPT and return the assistant message content.

    Requires OPENAI_API_KEY to be set in the environment. Does not print or log secrets.
    """
    # Ensure API key is available (the SDK will read from env automatically, but set explicitly if missing)
    if not getattr(openai, "api_key", None):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if timeout is not None:
        # For openai<=0.x the argument name is request_timeout
        kwargs["request_timeout"] = timeout

    # Prefer the newer SDK if available; fall back to legacy ChatCompletion API
    if _OpenAIClient is not None:  # New SDK path
        client = _OpenAIClient()
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    else:  # Legacy SDK path
        resp = openai.ChatCompletion.create(**kwargs)
        choice = resp["choices"][0]
        message = choice.get("message") or {}
        return (message.get("content") or "")

start = time.perf_counter()
with ThreadPoolExecutor() as pool:
    futures = []
    for i in range(10):
        future = pool.submit(prompt_chatgpt, f'hello world, I am currently thread {i}')
        futures.append(future)
for r in futures:
    print(r.result())
end_time = time.perf_counter()
print(f'finished running all the experiments; spent {end_time - start} seconds')
print('\n')

start = time.perf_counter()
for i in range(10):
    result = prompt_chatgpt(f'hello world, I am currently thread {i}')
    print(result)

end_time = time.perf_counter()
print(f'finished running all the experiments; spent {end_time - start} seconds')
print('\n')

start = time.perf_counter()

async def main():
    return await asyncio.gather(
        *[asyncio.to_thread(prompt_chatgpt, f'hello world, I am currently thread {i}') for i in range(10)]
    )

results = asyncio.run(main())
for r in futures:
    print(r.result())

end_time = time.perf_counter()
print(f'finished running all the experiments; spent {end_time - start} seconds')
