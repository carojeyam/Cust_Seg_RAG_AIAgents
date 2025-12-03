import os
from typing import Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class LLMProvider:
    def generate(self, prompt: str, context: str = "") -> str:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "mistral"):
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama not installed. Install: pip install ollama")
        self.model = model

    def generate(self, prompt: str, context: str = "") -> str:
        try:
            full_prompt = f"{context}\n\nQuestion: {prompt}" if context else prompt
            # Optional: truncate long context
            full_prompt = full_prompt[:15000]
            response = ollama.generate(model=self.model, prompt=full_prompt, stream=False)
            return response.get("response", "No response generated").strip()
        except Exception as e:
            return f"Error: {e}"


class GroqProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "mixtral-8x7b-32768"):
        if not GROQ_AVAILABLE:
            raise ImportError("Groq not installed. Install: pip install groq")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set.")
        self.client = Groq(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, context: str = "") -> str:
        try:
            full_prompt = f"{context}\n\nQuestion: {prompt}" if context else prompt
            full_prompt = full_prompt[:15000]
            message = self.client.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model=self.model,
                max_tokens=1024,
                temperature=0.7
            )
            return message.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"


# Global LLM provider
_llm_provider: Optional[LLMProvider] = None

def get_llm_provider(provider: str = "ollama", **kwargs) -> Optional[LLMProvider]:
    if provider.lower() == "ollama" and OLLAMA_AVAILABLE:
        return OllamaProvider(**kwargs)
    elif provider.lower() == "groq" and GROQ_AVAILABLE:
        return GroqProvider(**kwargs)
    return None

def set_llm_provider(provider: str, **kwargs):
    global _llm_provider
    _llm_provider = get_llm_provider(provider, **kwargs)
    if _llm_provider:
        print(f"✓ LLM Provider enabled: {provider}")
    else:
        print(f"✗ Failed to initialize {provider} provider")

def enhance_with_llm(search_results: list, query: str) -> str:
    if not _llm_provider:
        return "\n\n".join(search_results)
    context = "Here is relevant information:\n\n" + "\n\n".join(search_results)
    return _llm_provider.generate(query, context)

def is_llm_enabled() -> bool:
    return _llm_provider is not None

def disable_llm():
    global _llm_provider
    _llm_provider = None
    print("✓ LLM disabled")
