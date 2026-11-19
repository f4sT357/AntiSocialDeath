import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DevelopmentLogger:
    """Opt-in JSONL diagnostics. Content-bearing fields are never persisted."""

    _SENSITIVE_KEYS = frozenset({
        "text", "prompt", "content", "original", "rewrite", "response", "messages", "input",
    })

    def __init__(self, path: Path, enabled: bool = False) -> None:
        self.path = path
        self.enabled = enabled

    def log(self, event: str, **details: Any) -> None:
        if not self.enabled:
            return
        safe_details = {
            key: value for key, value in details.items()
            if key.casefold() not in self._SENSITIVE_KEYS
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **safe_details}
        with self.path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def read_recent(self, limit: int = 100) -> tuple[list[dict[str, Any]], str]:
        if not self.path.exists():
            return [], ""
        text = self.path.read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        return records[-limit:], text

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
