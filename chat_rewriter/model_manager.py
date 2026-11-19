"""Local model discovery and loading through LM Studio's native REST API."""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
from typing import Any

from .llm import ModelSettings


class LMStudioAPIError(RuntimeError):
    """Raised when LM Studio model management is unavailable or rejects a request."""


class LMStudioModelManager:
    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings
        self.api_root = self._api_root(settings.api_base)

    @staticmethod
    def _api_root(api_base: str) -> str:
        parsed = urlsplit(api_base.rstrip("/"))
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))

    def _request(self, method: str, endpoint: str,
                 payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        api_key = self.settings.api_key.strip()
        if api_key and api_key.casefold() != "lm-studio":
            headers["Authorization"] = f"Bearer {api_key}"
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")

        request = Request(f"{self.api_root}/api/v1/{endpoint.lstrip('/')}",
                          data=body, headers=headers, method=method)
        timeout = 600 if endpoint.rstrip("/") == "models/load" else 15
        try:
            with urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LMStudioAPIError(
                f"LM Studio API returned HTTP {exc.code}: {detail or exc.reason}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise LMStudioAPIError(
                f"LM Studioのモデル管理APIに接続できません ({type(exc).__name__})") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LMStudioAPIError("LM Studio APIからJSON以外の応答が返りました。") from exc

        if not isinstance(data, dict):
            raise LMStudioAPIError("LM Studio APIの応答形式が不正です。")
        return data

    def list_models(self) -> list[dict[str, Any]]:
        response = self._request("GET", "models")
        models = response.get("models")
        if not isinstance(models, list):
            raise LMStudioAPIError("LM Studioのモデル一覧にmodels配列がありません。")
        return [model for model in models if isinstance(model, dict)]

    def ensure_loaded(self, model_key: str) -> dict[str, Any]:
        model = next((item for item in self.list_models()
                      if item.get("key") == model_key and item.get("type") == "llm"), None)
        if model is None:
            raise LMStudioAPIError(
                f"モデル {model_key} がLM Studioのローカルモデル一覧にありません。"
                "LM Studioにダウンロードしてから、モデル一覧を更新してください。")
        if model.get("loaded_instances"):
            return {"status": "already_loaded", "model": model_key, "load_time_seconds": 0}
        max_context = model.get("max_context_length")
        context_length = min(8192, max_context) if isinstance(max_context, int) and max_context > 0 else 8192
        return self._request("POST", "models/load", {
            "model": model_key,
            "context_length": context_length,
        })
