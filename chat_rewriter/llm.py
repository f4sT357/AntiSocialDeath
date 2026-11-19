import json
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from json_repair import repair_json
from openai import OpenAI

from .config import MAX_JSON_REPAIR_ATTEMPTS
from .diagnostics import DevelopmentLogger

CallStage = Literal["analysis", "rewrite", "jev"]
JSON_REPAIR_INSTRUCTION = (
    "前の回答はJSONとして読み取れませんでした。説明文やMarkdownを付けず、"
    "有効なJSONオブジェクトだけを返してください。"
)


@dataclass(frozen=True)
class ModelSettings:
    api_base: str
    api_key: str
    model: str
    timeout_seconds: int = 90


class IncompleteModelResponseError(ValueError):
    """Raised when the model spends its output budget before returning content."""


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse model JSON, tolerating wrappers and common syntax mistakes."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    repaired = repair_json(cleaned, return_objects=True, skip_json_loads=True)
    if isinstance(repaired, dict):
        return repaired
    raise ValueError("モデルの応答からJSONオブジェクトを読み取れませんでした。")


def validate_payload(stage: CallStage, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate stage contracts after parsing or repairing JSON."""
    if stage == "analysis":
        if not isinstance(payload.get("risks", []), list):
            raise ValueError("分析結果のrisksは配列である必要があります。")
        if not isinstance(payload.get("emotions", {}), dict):
            raise ValueError("分析結果のemotionsはオブジェクトである必要があります。")
    elif stage == "rewrite":
        rewrite = payload.get("rewrite")
        if not isinstance(rewrite, str) or not rewrite.strip():
            raise ValueError("リライト結果に有効なrewrite文字列がありません。")
    elif stage == "jev":
        if not isinstance(payload.get("pass"), bool):
            raise ValueError("Jev結果に真偽値のpass判定がありません。")
        if not isinstance(payload.get("issues", []), list):
            raise ValueError("Jev結果のissuesは配列である必要があります。")
    return payload


class StructuredLLM:
    def __init__(self, settings: ModelSettings, logger: DevelopmentLogger) -> None:
        self.settings = settings
        self.logger = logger
        self.client = OpenAI(base_url=settings.api_base.rstrip("/"),
                             api_key=settings.api_key or "lm-studio",
                             timeout=settings.timeout_seconds, max_retries=0)

    def list_models(self) -> list[str]:
        """Return chat-capable model IDs, excluding embedding-only models."""
        models = self.client.models.list().data
        return [model.id for model in models if "embed" not in model.id.casefold()]

    def check_connection(self) -> int:
        return len(self.list_models())

    def complete(self, stage: CallStage, system: str, user: str) -> dict[str, Any]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        max_tokens = 500 if stage == "jev" else 1400
        for attempt in range(MAX_JSON_REPAIR_ATTEMPTS + 1):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.model or "local-model",
                    temperature=0.1,
                    max_tokens=max_tokens,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "structured_response", "strict": False,
                        "schema": {"type": "object", "additionalProperties": True}}},
                    # Current llama.cpp-backed LM Studio ignores chat_template_kwargs
                    # for some reasoning models. Its OpenAI-compatible API supports
                    # reasoning_effort="none" to request a direct answer.
                    extra_body={"reasoning_effort": "none"},
                    messages=messages,
                )
            except Exception as exc:
                self.logger.log("llm_error", stage=stage, attempt=attempt + 1,
                                model=self.settings.model,
                                elapsed_ms=self._elapsed_ms(started), error_type=type(exc).__name__)
                raise

            message = response.choices[0].message
            content = message.content or ""
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length" and not content.strip():
                self.logger.log("model_output_truncated", stage=stage, attempt=attempt + 1,
                                model=self.settings.model,
                                elapsed_ms=self._elapsed_ms(started),
                                reasoning_tokens=getattr(getattr(response.usage, "completion_tokens_details", None),
                                                         "reasoning_tokens", None))
                guidance = (
                    f"{self.settings.model} が出力上限を内部推論で使い切り、本文を返しませんでした。"
                    "LM Studioの推論設定で思考/ReasoningをOFFにするか、非推論モデルを選択してください。"
                )
                raise IncompleteModelResponseError(guidance)

            try:
                payload = validate_payload(stage, parse_json_object(content))
            except ValueError as exc:
                self.logger.log("invalid_model_json", stage=stage, attempt=attempt + 1,
                                model=self.settings.model,
                                elapsed_ms=self._elapsed_ms(started), error_type=type(exc).__name__)
                if attempt >= MAX_JSON_REPAIR_ATTEMPTS:
                    raise
                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": JSON_REPAIR_INSTRUCTION},
                ])
                continue

            usage = response.usage
            self.logger.log("llm_response", stage=stage, attempt=attempt + 1,
                            elapsed_ms=self._elapsed_ms(started),
                            model=self.settings.model,
                            finish_reason=finish_reason,
                            prompt_tokens=getattr(usage, "prompt_tokens", None),
                            completion_tokens=getattr(usage, "completion_tokens", None))
            return payload
        raise ValueError("モデルの応答をJSONとして読み取れませんでした。")

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return round((time.perf_counter() - started) * 1000)
