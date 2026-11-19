from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
HISTORY_DB_PATH = DATA_DIR / "history.sqlite3"
DEVELOPMENT_LOG_PATH = DATA_DIR / "development.jsonl"

MAX_INPUT_CHARACTERS = 4999
DEFAULT_API_BASE = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_MODEL = "qwen3.5-4b"
DEFAULT_REWRITE_STRENGTH = "標準"
MAX_JSON_REPAIR_ATTEMPTS = 1
MAX_JEV_REWRITE_ATTEMPTS = 3

RISK_CATEGORIES = (
    "侮辱・人格攻撃", "名誉毀損・信用毀損", "攻撃性・威圧", "差別・偏見",
    "プライバシー侵害", "情報漏洩", "性的ハラスメント", "虚偽・不確かな情報",
    "対人関係リスク", "TPO不適合",
)
EMOTIONS = ("喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼")
NEGATIVE_EMOTIONS = frozenset(("悲しみ", "怒り", "恐れ", "嫌悪"))
