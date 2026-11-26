import os
from huggingface_hub import InferenceClient

HF_TOKEN = "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"

MODELS = ["openai/gpt-oss-120b", "Qwen/Qwen2.5-7B-Instruct",
           "MiniMax/MiniMax-M1-80K", "deepseek-ai/DeepSeek-V3",
            "meta-llama/Llama-3-8B-Instruct"]

def generate_response(prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
    key = HF_TOKEN
    if not key:
        return "Error: HF_TOKEN is missing in environment variables"

    last_err = None
    for m in MODELS:
        try:
            c = InferenceClient(model=m, token=key)
            r = c.chat.completions.create(
                messages=[{'role': 'user', 'content': prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return r.choices[0].message.content
        
        except Exception as e:
            last_err = e

    return (
        f"Hugging Face model failed.\n"
        f"Tried models: {MODELS}\n"
        f"Details: {type(last_err).__name__}: {last_err}"
    )
