import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from langchain_openai import ChatOpenAI


# ================= LLM backends =================
# A backend ("provider") is a function that turns an LLMSpec into a LangChain chat model.
# Built-in providers:
#   openai : any hosted OpenAI-compatible API. Reads API_KEY and Base_URL from the environment
#            (.env) unless the spec overrides them. This is the original behaviour.
#   ollama : a local Ollama server (e.g. qwen3:8b). Uses Ollama's OpenAI-compatible /v1 endpoint,
#            so no API key is needed.
# To add another backend (Anthropic, vLLM, ...), write a builder and decorate it with
# @register_provider("name"). Nothing else in the evaluation scripts has to change.

@dataclass(frozen=True)
class LLMSpec:
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    timeout: float = 120
    base_url: Optional[str] = None          # None -> provider default
    reasoning_effort: Optional[str] = None  # e.g. "none" turns off Qwen3 thinking on Ollama


PROVIDERS: Dict[str, Callable[[LLMSpec], object]] = {}


def register_provider(name):
    def decorator(builder):
        PROVIDERS[name] = builder
        return builder
    return decorator


def _extra_body(spec):
    return {"reasoning_effort": spec.reasoning_effort} if spec.reasoning_effort else None


def default_ollama_base_url():
    host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    if "://" not in host:
        host = f"http://{host}"
    return f"{host.rstrip('/')}/v1"


def resolve_endpoint(spec):
    """(base_url, api_key) for the OpenAI-compatible providers. Also used by memory systems
    (e.g. Mem0) that create their own client and need the same endpoint."""
    if spec.provider == "openai":
        return spec.base_url or os.getenv("Base_URL"), os.getenv("API_KEY")
    if spec.provider == "ollama":
        # The key is ignored by Ollama, but the OpenAI client requires a non-empty value.
        return spec.base_url or default_ollama_base_url(), "ollama"
    raise ValueError(f"Provider {spec.provider!r} has no OpenAI-compatible endpoint registered.")


def _build_openai_compatible(spec):
    base_url, api_key = resolve_endpoint(spec)
    return ChatOpenAI(
        model=spec.model,
        temperature=spec.temperature,
        api_key=api_key,
        base_url=base_url,
        timeout=spec.timeout,
        extra_body=_extra_body(spec),
    )


register_provider("openai")(_build_openai_compatible)
register_provider("ollama")(_build_openai_compatible)


def build_chat_model(spec):
    if spec.provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider {spec.provider!r}. Available: {sorted(PROVIDERS)}")
    return PROVIDERS[spec.provider](spec)


# Defaults match the original script: gpt-4o, warm agent, cold judge.
DEFAULT_AGENT_SPEC = LLMSpec(temperature=0.7, timeout=120)
DEFAULT_JUDGE_SPEC = LLMSpec(temperature=0.2, timeout=60)

_specs = {"agent": DEFAULT_AGENT_SPEC, "judge": DEFAULT_JUDGE_SPEC}
_models = {}


def configure_llms(agent_spec=None, judge_spec=None):
    """Choose the backends used by get_agent_response / get_asr_agent_response / get_judge_response.
    Call before the first request. Models are built lazily, so env vars can be loaded first."""
    if agent_spec is not None:
        _specs["agent"] = agent_spec
    if judge_spec is not None:
        _specs["judge"] = judge_spec
    _models.clear()


def get_llm_spec(role):
    """The LLMSpec currently configured for "agent" or "judge"."""
    return _specs[role]


def _get_model(role):
    if role not in _models:
        _models[role] = build_chat_model(_specs[role])
    return _models[role]


def check_llms_ready():
    """Fail fast if a backend is unreachable. get_*_response swallow errors and return a fallback,
    which would otherwise show up as silently wrong metrics."""
    for role in ("agent", "judge"):
        spec = _specs[role]
        try:
            _get_model(role).invoke("Reply with OK.")
        except Exception as e:
            raise RuntimeError(
                f"{role} LLM is not reachable (provider={spec.provider}, model={spec.model}, "
                f"base_url={spec.base_url or 'default'}): {e}"
            ) from e


def _content_text(response):
    content = response.content
    if isinstance(content, list):  # some providers return a list of content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content


# Qwen3 / DeepSeek-R1 style models may put reasoning in <think> tags inside the content
# (Ollama strips it into a separate field, other servers may not).
def clean_deepseek_thought(text):
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


def get_agent_response(prompt, max_retries=3):
    """
    Agent response generation with automatic retries to prevent the evaluation from being interrupted by timeouts.
    """
    for attempt in range(max_retries):
        try:
            response = _get_model("agent").invoke(prompt)
            # Must return a plain text string; never return response directly!
            return clean_deepseek_thought(_content_text(response))

        except Exception as e:
            print(f"[Warning] In get_agent_response (Attempt {attempt + 1}/{max_retries}): Error: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Wait 1s for the first retry, 2s for the second...
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("All retries failed. Returning default fallback.")
                return "I don't know"


def get_asr_agent_response(prompt, max_retries=3):
    """
    Agent response generation with automatic retries to prevent the evaluation from being interrupted by timeouts.
    """
    for attempt in range(max_retries):
        try:
            response = _get_model("agent").invoke(prompt)
            # Must return a plain text string; never return response directly!
            return clean_deepseek_thought(_content_text(response))

        except Exception as e:
            print(f"[Warning] In get_asr_agent_response (Attempt {attempt + 1}/{max_retries}): Error: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Wait 1s for the first retry, 2s for the second...
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("All retries failed. Returning default fallback.")
                return "I don't know"


def get_judge_response(original_text, verification_question, agent_answer, max_retries=3):
    prompt = f"""
    ### Task: Precise auditing of long-term conversational memory
    You are a professional academic evaluator responsible for determining whether an AI Agent accurately extracts and retains the key facts from the user's memory.

    ---

    ### [Core Data]
    1. Source Fact: "{original_text}"
    2. Verification Question: "{verification_question}"
    3. Agent Output: "{agent_answer}"

    ---
    ### [Evaluation Criteria]
 
    Criterion 1: Refusal-to-answer is scored as 0
    - If the answer includes "I don't know", "not mentioned", "I don't remember", or "cannot answer based on memory", you must score it as 0.

    Criterion 2: Semantic equivalence (core rule)
    - ✅ Score 1 if: the agent's answer semantically covers the key point in the source fact relevant to the question.
    - Reasonable paraphrasing is allowed. For example, if the original says "keep the rule of not eating peanut oil", and the answer says "only eat it if there is no peanut oil", this is considered an accurate semantic reconstruction.
    - Reasonable inference is allowed as long as the answer does not alter the original fact.

    Criterion 3: Factual errors are scored as 0
    - ❌ Score 0 if:
    - Key entity errors (e.g., remembering "peanut oil" as "sesame oil").
    - Completely opposite logic (e.g., remembering it as "likes peanut oil").
    - Hallucination interference: introducing external information that contradicts the source fact or was not present in the source.

    ---

    ### [Judging Requirements]
    Ignore filler words, tense differences, and non-essential modifiers.
    We only care about: does the agent capture the key knowledge point?

    ### [Output Requirements]
    Do NOT output your chain-of-thought. Only return a number:
    Judgment: [1 or 0]
    """

    for attempt in range(max_retries):
        try:
            response = _get_model("judge").invoke(prompt)
            content = clean_deepseek_thought(_content_text(response))

            # Minimal decision rule: return 1 if the output contains "1"; otherwise return 0
            if "1" in content:
                return "1"
            else:
                return "0"

        except Exception as e:
            print(f"\n[Warning] Error in ACC Judge (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = 3 ** attempt  # Wait 1s for the first retry, 3s for the second, 9s for the third
                print(f"Rate limited or Network error. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("All retries failed. Returning default 0.")
                return "0"
