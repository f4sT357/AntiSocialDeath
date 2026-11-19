import json
import time
from datetime import datetime
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import EMOTIONS, MAX_JEV_REWRITE_ATTEMPTS, NEGATIVE_EMOTIONS, RISK_CATEGORIES
from .diagnostics import DevelopmentLogger
from .llm import StructuredLLM

ANALYSIS_SYSTEM = (
    "あなたは日本語短文の社会的リスクと文章上の感情を評価するアシスタントです。"
    "法律判断や心理診断をせず、文章中の根拠に基づいて控えめに評価します。"
    "リスクは該当項目だけを返し、スコアは0から100。"
    "感情はPlutchikの8感情を0から3で推定します。JSONのみ返してください。"
)
REWRITE_SYSTEM = (
    "あなたは短文の最小限リライターです。原文の要求・不満・主張・事実を維持し、"
    "侮辱・人格攻撃だけを優先して除去し、不要な威圧や挑発を調整します。"
    "原文全体を別の文章に作り直さず、問題のある箇所だけを直してください。"
    "特に肯定/否定、賛成/反対、必要/不要、推奨する行動の方向、因果、時間軸、"
    "発言者・引用の帰属を絶対に反転・変更・混同しないでください。"
    "原文の意図が曖昧な箇所は、推測で補わず極力原文を残してください。"
    "怒りや明確な拒否は不当に弱めません。情報を追加しません。"
    "JSON {\"rewrite\":\"...\"} のみを返してください。"
)
JEV_SYSTEM = (
    "あなたは独立した品質検査役Jevです。原文と候補を文ごと・話者ごとに照合してください。"
    "侮辱除去に加え、すべての重要な主張が維持されていることを確認します。"
    "肯定/否定、賛成/反対、必要/不要、推奨する行動の方向、因果、時間軸、"
    "数値、事実、話者・引用の帰属の反転・変更・混同は重大な不合格理由です。"
    "重要な主張の欠落や、原文にない主張の追加も不合格にしてください。"
    "これらをすべて確認できた場合のみpass=trueにしてください。"
    "JSONで返す: {\"pass\":true,\"issues\":[\"...\"],\"feedback\":\"修正指示\"}. "
    "感情スコアを合否に使わない。"
)


class RewriteLoopState(TypedDict):
    original: str
    emotions: dict[str, int]
    instruction: str
    strength: str
    feedback: str
    candidate: str
    attempt: int
    passed: bool
    issues: list[Any]
    verdict: dict[str, Any]


def _bounded_int(value: Any, default: int, upper: int) -> int:
    try:
        return max(0, min(upper, int(value)))
    except (TypeError, ValueError):
        return default


def normalize_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    risks = payload.get("risks", [])
    clean_risks = []
    for item in risks if isinstance(risks, list) else []:
        if not isinstance(item, dict) or item.get("category") not in RISK_CATEGORIES:
            continue
        clean_risks.append({
            "category": item["category"],
            "score": _bounded_int(item.get("score"), 0, 100),
            "evidence": str(item.get("evidence", ""))[:300],
            "explanation": str(item.get("explanation", ""))[:500],
        })

    raw_emotions = payload.get("emotions", {})
    raw_emotions = raw_emotions if isinstance(raw_emotions, dict) else {}
    emotions = {name: _bounded_int(raw_emotions.get(name), 0, 3) for name in EMOTIONS}
    return {
        "risks": clean_risks,
        "emotions": emotions,
        "confidence": _bounded_int(payload.get("confidence"), 50, 100),
        "summary": str(payload.get("summary", ""))[:1000],
    }


class RewriteService:
    def __init__(self, llm: StructuredLLM, logger: DevelopmentLogger,
                 rewrite_strength: str, decision_llm: StructuredLLM | None = None) -> None:
        self.llm = llm
        self.decision_llm = decision_llm or llm
        self.logger = logger
        self.rewrite_strength = rewrite_strength
        self.rewrite_graph = self._build_rewrite_graph()

    def _build_rewrite_graph(self):
        graph = StateGraph(RewriteLoopState)
        graph.add_node("rewrite", self._rewrite_node)
        graph.add_node("judge", self._judge_node)
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "judge")
        graph.add_conditional_edges(
            "judge", self._route_after_judgement,
            {"rewrite": "rewrite", "end": END},
        )
        return graph.compile()

    def _rewrite_node(self, state: RewriteLoopState) -> dict[str, Any]:
        attempt = state["attempt"] + 1
        payload = {
            "original": state["original"],
            "negative_emotions": {key: score for key, score in state["emotions"].items()
                                  if key in NEGATIVE_EMOTIONS},
            "strength": state["strength"],
            "extra_instruction": state["instruction"],
            "jev_feedback": state["feedback"],
        }
        rewritten = self.llm.complete(
            "rewrite", REWRITE_SYSTEM, json.dumps(payload, ensure_ascii=False))
        candidate = rewritten["rewrite"].strip()
        self.logger.log("rewrite_loop_candidate", attempt=attempt)
        return {"candidate": candidate, "attempt": attempt}

    def _judge_node(self, state: RewriteLoopState) -> dict[str, Any]:
        verdict = self.decision_llm.complete(
            "jev", JEV_SYSTEM,
            json.dumps({
                "original": state["original"],
                "candidate": state["candidate"],
                "strength": state["strength"],
                "user_instruction": state["instruction"],
            }, ensure_ascii=False),
        )
        issues = verdict.get("issues", [])
        self.logger.log("rewrite_loop_judgement", attempt=state["attempt"],
                        passed=verdict["pass"], issue_count=len(issues))
        return {
            "verdict": verdict,
            "passed": verdict["pass"],
            "issues": issues,
            "feedback": str(verdict.get("feedback", "問題箇所を修正してください"))[:1000],
        }

    def _route_after_judgement(self, state: RewriteLoopState) -> Literal["rewrite", "end"]:
        if state["passed"] or state["attempt"] >= MAX_JEV_REWRITE_ATTEMPTS:
            if not state["passed"]:
                self.logger.log("rewrite_loop_limit_reached", attempts=state["attempt"])
            return "end"
        return "rewrite"

    def analyze(self, text: str) -> dict[str, Any]:
        schema = {
            "risks": [{"category": "分類名", "score": 0, "evidence": "該当箇所", "explanation": "説明"}],
            "emotions": {name: 0 for name in EMOTIONS}, "confidence": 50, "summary": "短い所見",
        }
        user = (
            f"分類候補: {json.dumps(RISK_CATEGORIES, ensure_ascii=False)}\n"
            f"出力形式例: {json.dumps(schema, ensure_ascii=False)}\n\n対象文:\n{text}"
        )
        return normalize_analysis(self.llm.complete("analysis", ANALYSIS_SYSTEM, user))

    def rewrite(self, text: str, emotions: dict[str, int], instruction: str = "") -> tuple[str, dict[str, Any]]:
        state = self.rewrite_graph.invoke({
            "original": text,
            "emotions": emotions,
            "instruction": instruction,
            "strength": self.rewrite_strength,
            "feedback": "",
            "candidate": "",
            "attempt": 0,
            "passed": False,
            "issues": [],
            "verdict": {},
        }, config={"recursion_limit": MAX_JEV_REWRITE_ATTEMPTS * 2 + 1})
        result = {"passed": state["passed"], "attempts": state["attempt"],
                  "issues": state["issues"]}
        if not state["passed"]:
            result["unresolved"] = state["feedback"]
        return state["candidate"], result

    def run(self, original: str, instruction: str = "") -> dict[str, Any]:
        started = time.perf_counter()
        self.logger.log("pipeline_started", character_count=len(original),
                        has_extra_instruction=bool(instruction))
        try:
            before = self.analyze(original)
            rewritten, jev = self.rewrite(original, before["emotions"], instruction)
            after = self.analyze(rewritten)
            self.logger.log("pipeline_completed", elapsed_ms=self._elapsed_ms(started),
                            jev_passed=jev["passed"], jev_attempts=jev["attempts"])
            return {"original": original, "rewrite": rewritten, "before": before,
                    "after": after, "jev": jev,
                    "created_at": datetime.now().astimezone().isoformat(timespec="seconds")}
        except Exception as exc:
            self.logger.log("pipeline_error", elapsed_ms=self._elapsed_ms(started),
                            error_type=type(exc).__name__)
            raise

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return round((time.perf_counter() - started) * 1000)
