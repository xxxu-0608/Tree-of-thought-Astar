import os
import openai
import backoff 

completion_tokens = prompt_tokens = 0
openai.api_base="https://hkust.azure-api.net"
openai.api_key="6bd9ca9f8ab14ed3a0080232bd09f897"
openai.api_type = "azure"
openai.api_version ="2024-10-21"

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-35-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-35-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(engine=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-35-turbo"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-35-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
