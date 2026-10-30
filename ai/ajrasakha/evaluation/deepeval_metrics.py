import os
import functools
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import AnthropicModel, OpenAIModel

# Load .env once at import
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

class JudgeSetupError(RuntimeError):
    pass

@dataclass
class JudgeConfig:
    provider: str
    model: str
    api_key: str = field(repr=False)
    base_url: Optional[str] = None

def load_judge_config() -> JudgeConfig:
    provider = os.getenv("JUDGE_PROVIDER", "anthropic").lower()
    model = os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    base_url = os.getenv("JUDGE_BASE_URL")
    
    if provider == "anthropic":
        api_key = os.getenv("JUDGE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    elif provider == "openai":
        api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    else:
        raise JudgeSetupError(f"Unknown JUDGE_PROVIDER='{provider}'. Use 'anthropic' or 'openai'.")
        
    if not api_key:
        raise JudgeSetupError(f"API_KEY not set. Add JUDGE_API_KEY (or {provider.upper()}_API_KEY) to ai/.env.")
        
    return JudgeConfig(provider=provider, model=model, base_url=base_url, api_key=api_key)

@functools.lru_cache(maxsize=1)
def get_judge_model():
    config = load_judge_config()
    kwargs = {"model": config.model, "api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
        
    if config.provider == "anthropic":
        return AnthropicModel(**kwargs)
    return OpenAIModel(**kwargs)

def _build_metric(metric_cls, model, threshold: float = 0.5):
    return metric_cls(threshold=threshold, model=model)

def _metric_passed(metric) -> bool:
    if hasattr(metric, "passed"):
        return bool(metric.passed)
    if hasattr(metric, "is_successful"):
        return bool(metric.is_successful())
    return False

def evaluate_answer_with_deepeval(
    query: str,
    answer: str,
    context: list[str] | None = None,
):
    context = context or []

    if not answer or not str(answer).strip():
        return {
            "AnswerRelevancyMetric": {"score": None, "passed": False, "reason": "answer_missing"},
            "FaithfulnessMetric": {"score": None, "passed": False, "reason": "answer_missing"},
            "ContextualRelevancyMetric": {"score": None, "passed": False, "reason": "answer_missing"},
        }

    judge_model = get_judge_model()
    test_case = LLMTestCase(input=query, actual_output=answer, retrieval_context=context)
    
    metrics = [
        _build_metric(AnswerRelevancyMetric, judge_model),
        _build_metric(FaithfulnessMetric, judge_model),
        _build_metric(ContextualRelevancyMetric, judge_model),
    ]

    results = {}
    for metric in metrics:
        metric_name = metric.__class__.__name__
        try:
            metric.measure(test_case)
            results[metric_name] = {
                "score": metric.score,
                "passed": _metric_passed(metric),
                "reason": metric.reason,
            }
        except Exception as e:
            results[metric_name] = {"score": None, "passed": False, "reason": str(e)}
    return results
