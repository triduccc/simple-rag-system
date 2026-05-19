from config import settings
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def _build_gemini():
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
    )

@lru_cache(maxsize=4)
def get_llm(provider=None):
    provider = provider or settings.llm_provider

    if provider == "gemini":
        return _build_gemini()

    raise ValueError(f"Unknown llm_provider ’{provider}’")

def invoke_llm(prompt, provider=None):
    response = get_llm(provider=provider).invoke([HumanMessage(content=prompt)])
    return response.content if isinstance(response.content, str) else str(response.content)
