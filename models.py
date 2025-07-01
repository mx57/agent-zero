from enum import Enum
import os
from typing import Any
from langchain_openai import (
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    AzureOpenAI,
)
from langchain_community.llms.ollama import Ollama
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.exceptions import LangChainException
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
    embeddings as google_embeddings,
)
from langchain_mistralai import ChatMistralAI

# from pydantic.v1.types import SecretStr
from python.helpers import dotenv, runtime
from python.helpers.dotenv import load_dotenv
from python.helpers.rate_limiter import RateLimiter

# environment variables
load_dotenv()


class ModelType(Enum):
    CHAT = "Chat"
    EMBEDDING = "Embedding"


class ModelProvider(Enum):
    ANTHROPIC = "Anthropic"
    CHUTES = "Chutes"
    DEEPSEEK = "DeepSeek"
    GOOGLE = "Google"
    GROQ = "Groq"
    HUGGINGFACE = "HuggingFace"
    LMSTUDIO = "LM Studio"
    MISTRALAI = "Mistral AI"
    OLLAMA = "Ollama"
    OPENAI = "OpenAI"
    OPENAI_AZURE = "OpenAI Azure"
    OPENROUTER = "OpenRouter"
    SAMBANOVA = "Sambanova"
    OTHER = "Other"


rate_limiters: dict[str, RateLimiter] = {}


# Utility function to get API keys from environment variables
def get_api_key(service):
    return (
        dotenv.get_dotenv_value(f"API_KEY_{service.upper()}")
        or dotenv.get_dotenv_value(f"{service.upper()}_API_KEY")
        or dotenv.get_dotenv_value(
            f"{service.upper()}_API_TOKEN"
        )  # Added for CHUTES_API_TOKEN
        or "None"
    )


def get_model(type: ModelType, provider: ModelProvider, name: str, **kwargs):
    fnc_name = f"get_{provider.name.lower()}_{type.name.lower()}"  # function name of model getter
    model = globals()[fnc_name](name, **kwargs)  # call function by name
    return model


def get_rate_limiter(
    provider: ModelProvider, name: str, requests: int, input: int, output: int
) -> RateLimiter:
    # get or create
    key = f"{provider.name}\\{name}"
    rate_limiters[key] = limiter = rate_limiters.get(key, RateLimiter(seconds=60))
    # always update
    limiter.limits["requests"] = requests or 0
    limiter.limits["input"] = input or 0
    limiter.limits["output"] = output or 0
    return limiter


def parse_chunk(chunk: Any):
    if isinstance(chunk, str):
        content = chunk
    elif hasattr(chunk, "content"):
        content = str(chunk.content)
    else:
        content = str(chunk)
    return content


# Ollama models
def get_ollama_base_url():
    base_url = dotenv.get_dotenv_value("OLLAMA_BASE_URL")
    if not base_url:
        default_url = f"http://{runtime.get_local_url()}:11434"
        print(f"OLLAMA_BASE_URL not set, using default: {default_url}")
        return default_url
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        print(f"Warning: OLLAMA_BASE_URL '{base_url}' does not start with http:// or https://. This might cause connection issues.")
    return base_url


def get_ollama_chat(
    model_name: str,
    base_url=None,
    num_ctx=8192,
    **kwargs,
):
    if not base_url:
        base_url = get_ollama_base_url()
    try:
        print(f"Initializing Ollama chat model: {model_name} at {base_url} with num_ctx: {num_ctx} and kwargs: {kwargs}")
        model = ChatOllama(
            model=model_name,
            base_url=base_url,
            num_ctx=num_ctx,
            **kwargs,
        )
        # Attempt a simple interaction to check connectivity if possible, or rely on langchain's internal checks.
        # For now, we assume initialization success means basic validity.
        print(f"Successfully initialized Ollama chat model: {model_name}")
        return model
    except LangChainException as e:
        print(f"Error initializing Ollama chat model {model_name} at {base_url}: {e}")
        # Potentially raise a custom, more user-friendly exception or handle specific common errors.
        # For example, check if 'Connection refused' is in str(e) to suggest Ollama server issues.
        if "Connection refused" in str(e):
            print(f"Hint: Ensure Ollama server is running and accessible at {base_url}.")
        elif "404" in str(e) and "model" in str(e).lower() and f"'{model_name}'" in str(e).lower(): # Heuristic for model not found
            print(f"Hint: Model '{model_name}' might not be available on the Ollama server or could be misspelled.")
        raise  # Re-raise the original exception after logging
    except Exception as e:
        print(f"An unexpected error occurred while initializing Ollama chat model {model_name}: {e}")
        raise


def get_ollama_embedding(
    model_name: str,
    base_url=None,
    num_ctx=8192,
    **kwargs,
):
    if not base_url:
        base_url = get_ollama_base_url()
    try:
        print(f"Initializing Ollama embedding model: {model_name} at {base_url} with num_ctx: {num_ctx} and kwargs: {kwargs}")
        embedding = OllamaEmbeddings(
            model=model_name, base_url=base_url, num_ctx=num_ctx, **kwargs
        )
        # Test embedding something simple
        embedding.embed_query("test")
        print(f"Successfully initialized Ollama embedding model: {model_name}")
        return embedding
    except LangChainException as e:
        print(f"Error initializing Ollama embedding model {model_name} at {base_url}: {e}")
        if "Connection refused" in str(e):
            print(f"Hint: Ensure Ollama server is running and accessible at {base_url}.")
        elif "404" in str(e) and "model" in str(e).lower() and f"'{model_name}'" in str(e).lower():
             print(f"Hint: Model '{model_name}' might not be available on the Ollama server or could be misspelled.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while initializing Ollama embedding model {model_name}: {e}")
        raise


# HuggingFace models
def get_huggingface_chat(
    model_name: str,
    api_key=None,
    **kwargs,
):
    # different naming convention here
    if not api_key:
        api_key = get_api_key("huggingface") or os.environ["HUGGINGFACEHUB_API_TOKEN"]

    # Initialize the HuggingFaceEndpoint with the specified model and parameters
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        do_sample=True,
        **kwargs,
    )

    # Initialize the ChatHuggingFace with the configured llm
    return ChatHuggingFace(llm=llm)


def get_huggingface_embedding(model_name: str, **kwargs):
    return HuggingFaceEmbeddings(model_name=model_name, **kwargs)


# LM Studio and other OpenAI compatible interfaces
def get_lmstudio_base_url():
    return (
        dotenv.get_dotenv_value("LM_STUDIO_BASE_URL")
        or f"http://{runtime.get_local_url()}:1234/v1"
    )


def get_lmstudio_chat(
    model_name: str,
    base_url=None,
    **kwargs,
):
    if not base_url:
        base_url = get_lmstudio_base_url()
    return ChatOpenAI(model_name=model_name, base_url=base_url, api_key="none", **kwargs)  # type: ignore


def get_lmstudio_embedding(
    model_name: str,
    base_url=None,
    **kwargs,
):
    if not base_url:
        base_url = get_lmstudio_base_url()
    return OpenAIEmbeddings(model=model_name, api_key="none", base_url=base_url, check_embedding_ctx_length=False, **kwargs)  # type: ignore


# Anthropic models
def get_anthropic_chat(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("anthropic")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        )
    return ChatAnthropic(model_name=model_name, api_key=api_key, base_url=base_url, **kwargs)  # type: ignore


# right now anthropic does not have embedding models, but that might change
def get_anthropic_embedding(
    model_name: str,
    api_key=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("anthropic")
    return OpenAIEmbeddings(model=model_name, api_key=api_key, **kwargs)  # type: ignore


# OpenAI models
def get_openai_chat(
    model_name: str,
    api_key=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("openai")
    return ChatOpenAI(model_name=model_name, api_key=api_key, **kwargs)  # type: ignore


def get_openai_embedding(model_name: str, api_key=None, **kwargs):
    if not api_key:
        api_key = get_api_key("openai")
    return OpenAIEmbeddings(model=model_name, api_key=api_key, **kwargs)  # type: ignore


def get_openai_azure_chat(
    deployment_name: str,
    api_key=None,
    azure_endpoint=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("openai_azure")
    if not azure_endpoint:
        azure_endpoint = dotenv.get_dotenv_value("OPENAI_AZURE_ENDPOINT")
    return AzureChatOpenAI(deployment_name=deployment_name, api_key=api_key, azure_endpoint=azure_endpoint, **kwargs)  # type: ignore


def get_openai_azure_embedding(
    deployment_name: str,
    api_key=None,
    azure_endpoint=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("openai_azure")
    if not azure_endpoint:
        azure_endpoint = dotenv.get_dotenv_value("OPENAI_AZURE_ENDPOINT")
    return AzureOpenAIEmbeddings(deployment_name=deployment_name, api_key=api_key, azure_endpoint=azure_endpoint, **kwargs)  # type: ignore


# Google models
def get_google_chat(
    model_name: str,
    api_key=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("google")
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}, **kwargs)  # type: ignore


def get_google_embedding(
    model_name: str,
    api_key=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("google")
    return google_embeddings.GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key, **kwargs)  # type: ignore


# Mistral models
def get_mistralai_chat(
    model_name: str,
    api_key=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("mistral")
    return ChatMistralAI(model=model_name, api_key=api_key, **kwargs)  # type: ignore


# Groq models
def get_groq_chat(
    model_name: str,
    api_key=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("groq")
    return ChatGroq(model_name=model_name, api_key=api_key, **kwargs)  # type: ignore


# DeepSeek models
def get_deepseek_chat(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("deepseek")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        )

    return ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url, **kwargs)  # type: ignore


# OpenRouter models
def get_openrouter_chat(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("openrouter")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("OPEN_ROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
    return ChatOpenAI(
        api_key=api_key, # type: ignore
        model=model_name,
        base_url=base_url,
        stream_usage=True,
        model_kwargs={
            "extra_headers": {
                "HTTP-Referer": "https://agent-zero.ai",
                "X-Title": "Agent Zero",
            }
        },
        **kwargs,
    )


def get_openrouter_embedding(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("openrouter")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("OPEN_ROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
    return OpenAIEmbeddings(model=model_name, api_key=api_key, base_url=base_url, **kwargs)  # type: ignore


# Sambanova models
def get_sambanova_chat(
    model_name: str,
    api_key=None,
    base_url=None,
    max_tokens=1024,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("sambanova")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("SAMBANOVA_BASE_URL")
            or "https://fast-api.snova.ai/v1"
        )
    return ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url, max_tokens=max_tokens, **kwargs)  # type: ignore


# right now sambanova does not have embedding models, but that might change
def get_sambanova_embedding(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("sambanova")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("SAMBANOVA_BASE_URL")
            or "https://fast-api.snova.ai/v1"
        )
    return OpenAIEmbeddings(model=model_name, api_key=api_key, base_url=base_url, **kwargs)  # type: ignore


# Other OpenAI compatible models
def get_other_chat(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    return ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url, **kwargs)  # type: ignore


def get_other_embedding(model_name: str, api_key=None, base_url=None, **kwargs):
    return OpenAIEmbeddings(model=model_name, api_key=api_key, base_url=base_url, **kwargs)  # type: ignore


# Chutes models
def get_chutes_chat(
    model_name: str,
    api_key=None,
    base_url=None,
    **kwargs,
):
    if not api_key:
        api_key = get_api_key("chutes")
    if not base_url:
        base_url = (
            dotenv.get_dotenv_value("CHUTES_BASE_URL") or "https://llm.chutes.ai/v1"
        )
    return ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url, **kwargs)  # type: ignore
