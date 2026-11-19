import csv
import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class HistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("""CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, original TEXT NOT NULL,
            result_json TEXT NOT NULL)""")
        return connection

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, created_at, original, result_json FROM history ORDER BY id DESC"
            ).fetchall()
        return [{"id": row[0], "created_at": row[1], "original": row[2],
                 "result": json.loads(row[3])} for row in rows]

    def add(self, original: str, result: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO history(created_at, original, result_json) VALUES (?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), original,
                 json.dumps(result, ensure_ascii=False)),
            )

    def delete(self, record_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM history WHERE id=?", (record_id,))

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM history")

    def export_json(self) -> str:
        return json.dumps(self.list(), ensure_ascii=False, indent=2)

    def export_csv(self) -> str:
        buffer = io.StringIO()
        fields = ["created_at", "original", "rewrite", "jev_passed"]
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for row in self.list():
            result = row["result"]
            writer.writerow({"created_at": row["created_at"], "original": row["original"],
                             "rewrite": result.get("rewrite", ""),
                             "jev_passed": result.get("jev", {}).get("passed", False)})
        return buffer.getvalue()
