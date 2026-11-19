import json
import html
import sqlite3
import time
from datetime import datetime
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from .config import (
    DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL, DEFAULT_REWRITE_STRENGTH,
    DEVELOPMENT_LOG_PATH, HISTORY_DB_PATH, MAX_INPUT_CHARACTERS,
    RISK_CATEGORIES,
)
from .diagnostics import DevelopmentLogger
from .llm import ModelSettings, StructuredLLM
from .model_manager import LMStudioAPIError, LMStudioModelManager
from .service import RewriteService
from .storage import HistoryStore

SAME_AS_INFERENCE = "推論モデルと同じ"


def _initialize_state() -> None:
    defaults = {
        "api_base": DEFAULT_API_BASE,
        "api_key": DEFAULT_API_KEY,
        "model": DEFAULT_MODEL,
        "decision_model": SAME_AS_INFERENCE,
        "decision_model_manual": "",
        "decision_model_auto_select": True,
        "save_history": False,
        "strength": DEFAULT_REWRITE_STRENGTH,
        "source": "",
        "result": None,
        "error": None,
        "dev_logging": False,
        "available_models": [],
        "model_statuses": {},
        "model_management_supported": None,
        "model_load_attempted": False,
        "model_load_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _make_logger() -> DevelopmentLogger:
    return DevelopmentLogger(DEVELOPMENT_LOG_PATH, enabled=st.session_state.dev_logging)


def _make_service(logger: DevelopmentLogger) -> RewriteService:
    shared_settings = {
        "api_base": st.session_state.api_base,
        "api_key": st.session_state.api_key,
    }
    inference_llm = StructuredLLM(
        ModelSettings(**shared_settings, model=st.session_state.model), logger)
    decision_model = st.session_state.decision_model
    if not st.session_state.available_models:
        decision_model = st.session_state.decision_model_manual.strip() or SAME_AS_INFERENCE
    decision_llm = inference_llm if decision_model == SAME_AS_INFERENCE else StructuredLLM(
        ModelSettings(**shared_settings, model=decision_model), logger)
    return RewriteService(inference_llm, logger, st.session_state.strength, decision_llm)


def _load_models(logger: DevelopmentLogger) -> list[str]:
    """Fetch local model IDs, including downloaded models that are not loaded yet."""
    settings = ModelSettings(
        api_base=st.session_state.api_base,
        api_key=st.session_state.api_key,
        model=st.session_state.model,
    )
    try:
        local_models = LMStudioModelManager(settings).list_models()
        llm_models = [model for model in local_models if model.get("type") == "llm"]
        models = [str(model["key"]) for model in llm_models if model.get("key")]
        st.session_state.model_statuses = {
            str(model["key"]): bool(model.get("loaded_instances"))
            for model in llm_models if model.get("key")
        }
        st.session_state.model_management_supported = True
    except LMStudioAPIError as exc:
        # Older LM Studio versions expose only loaded models through the OpenAI API.
        models = _make_service(logger).llm.list_models()
        st.session_state.model_statuses = {model: True for model in models}
        st.session_state.model_management_supported = False
        logger.log("model_management_api_unavailable", error_type=type(exc).__name__)

    st.session_state.available_models = models
    st.session_state.model_load_attempted = True
    st.session_state.model_load_error = None
    if models and st.session_state.model not in models:
        preferred_inference = next((model for model in models
                                    if "lfm2.5" not in model.casefold()
                                    and "lfm2-5" not in model.casefold()), None)
        st.session_state.model = (DEFAULT_MODEL if DEFAULT_MODEL in models
                                  else preferred_inference or models[0])
    if st.session_state.decision_model_auto_select:
        preferred = next((model for model in models
                          if "lfm2.5" in model.casefold() or "lfm2-5" in model.casefold()), None)
        if preferred:
            st.session_state.decision_model = preferred
            st.session_state.decision_model_auto_select = False
    return models


def _mark_decision_model_selected() -> None:
    st.session_state.decision_model_auto_select = False


def _ensure_selected_models_loaded(logger: DevelopmentLogger) -> None:
    """Load both selected chat models before inference when native management is available."""
    inference_model = st.session_state.model
    decision_model = st.session_state.decision_model
    if not st.session_state.available_models:
        decision_model = st.session_state.decision_model_manual.strip() or SAME_AS_INFERENCE
    selected = list(dict.fromkeys(
        model for model in (inference_model, decision_model)
        if model and model != SAME_AS_INFERENCE
    ))

    if not st.session_state.model_management_supported:
        unloaded = [model for model in selected
                    if not st.session_state.model_statuses.get(model, False)]
        if unloaded:
            raise LMStudioAPIError(
                "このLM Studioではモデル自動ロードAPIを使えません。"
                "LM Studio 0.4以降に更新するか、ロード済みモデルを選んでください。")
        return

    manager = LMStudioModelManager(ModelSettings(
        api_base=st.session_state.api_base,
        api_key=st.session_state.api_key,
        model=inference_model,
    ))
    for model in selected:
        started = time.perf_counter()
        try:
            result = manager.ensure_loaded(model)
        except Exception as exc:
            logger.log("model_load_error", model=model, error_type=type(exc).__name__)
            raise
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        loaded_now = result.get("status") == "loaded"
        st.session_state.model_statuses[model] = True
        logger.log("model_loaded" if loaded_now else "model_already_loaded",
                   model=model, elapsed_ms=elapsed_ms,
                   load_time_seconds=result.get("load_time_seconds", 0))


def _run_pipeline(original: str, instruction: str, logger: DevelopmentLogger) -> dict[str, Any]:
    _ensure_selected_models_loaded(logger)
    return _make_service(logger).run(original, instruction)


def _render_development_log_controls(logger: DevelopmentLogger) -> None:
    if not DEVELOPMENT_LOG_PATH.exists():
        return
    st.download_button(
        "開発ログをダウンロード", DEVELOPMENT_LOG_PATH.read_bytes(),
        "chat_rewriter_development.jsonl", "application/x-ndjson",
    )


def _format_log_record(record: dict[str, Any]) -> str:
    raw_time = str(record.get("time", ""))
    try:
        timestamp = datetime.fromisoformat(raw_time).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        timestamp = raw_time or "時刻不明"

    event = str(record.get("event", "event"))
    level = "ERROR" if "error" in event else "INFO"
    details = " ".join(f"{key}={value}" for key, value in record.items()
                       if key not in {"time", "event"})
    return f"{timestamp} [{level}] {event}" + (f" | {details}" if details else "")


def _render_development_log_view(logger: DevelopmentLogger) -> None:
    if not DEVELOPMENT_LOG_PATH.exists():
        return
    try:
        records, raw_text = logger.read_recent(limit=100)
        total_count = sum(1 for line in raw_text.splitlines() if line.strip())
        st.subheader("ログ")
        st.caption(f"{total_count} 件中、最新{len(records)}件")
        lines = "\n".join(_format_log_record(item) for item in records)
        safe_lines = html.escape(lines or "ログはまだありません。")
        components.html(
            f"""<div id="log-panel"><pre>{safe_lines}</pre></div>
            <style>
              html, body {{ margin: 0; padding: 0; background: transparent; }}
              #log-panel {{ height: 280px; overflow-y: auto; overflow-x: auto;
                box-sizing: border-box; background: #292d2f; color: #66ff20;
                border: 1px solid #666; padding: 8px 12px; font: 12px/1.5 Consolas, monospace; }}
              #log-panel pre {{ margin: 0; white-space: pre; }}
            </style>
            <script>
              const panel = document.getElementById('log-panel');
              panel.scrollTop = panel.scrollHeight;
            </script>""",
            height=296,
            scrolling=False,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        st.warning(f"ログを表示できません: {type(exc).__name__}")


def _render_history(store: HistoryStore) -> None:
    st.subheader("履歴")
    if not HISTORY_DB_PATH.exists():
        return
    try:
        rows = store.list()
        query = st.text_input("キーワード検索")
        category = st.selectbox("リスク分類で絞り込み", ["すべて", *RISK_CATEGORIES])
        visible = []
        for row in rows:
            result = row["result"]
            if query and query.casefold() not in row["original"].casefold() and query.casefold() not in result.get("rewrite", "").casefold():
                continue
            before_risks = result.get("before", {}).get("risks", [])
            if category != "すべて" and not any(item.get("category") == category for item in before_risks):
                continue
            visible.append(row)

        for row in visible[:30]:
            title = f"{row['created_at'][:16]} — {row['original'][:24]}"
            with st.expander(title):
                st.write(row["result"].get("rewrite", ""))
                if st.button("この履歴を削除", key=f"delete_history_{row['id']}"):
                    store.delete(row["id"])
                    st.rerun()

        if rows:
            st.download_button("JSONエクスポート", store.export_json(),
                               "chat_rewriter_history.json", "application/json")
            st.download_button("CSVエクスポート", store.export_csv(),
                               "chat_rewriter_history.csv", "text/csv")
            if st.button("全履歴を削除"):
                store.clear()
                st.rerun()
    except (sqlite3.Error, json.JSONDecodeError, OSError) as exc:
        st.error(f"履歴を読み込めません: {type(exc).__name__}")


def _render_sidebar(logger: DevelopmentLogger, store: HistoryStore) -> None:
    with st.sidebar:
        st.header("設定")
        st.text_input("LM Studio API URL", key="api_base", help="例: http://localhost:1234/v1")
        st.text_input("APIキー（必要な場合）", key="api_key", type="password")
        if st.session_state.available_models:
            def model_label(model_id: str) -> str:
                state = "ロード済み" if st.session_state.model_statuses.get(model_id) else "未ロード・実行時にロード"
                return f"{model_id}  [{state}]"

            st.selectbox("推論モデル", st.session_state.available_models,
                         key="model", format_func=model_label)
            decision_options = [SAME_AS_INFERENCE, *st.session_state.available_models]
            if st.session_state.decision_model not in decision_options:
                st.session_state.decision_model = SAME_AS_INFERENCE
            st.selectbox(
                "判断モデル（Jev）", decision_options, key="decision_model",
                on_change=_mark_decision_model_selected,
                format_func=lambda model_id: (model_id if model_id == SAME_AS_INFERENCE
                                              else model_label(model_id)),
                help="原文とリライト案の品質検査に使います。推論モデルとは別のモデルを指定できます。",
            )
            st.caption("推論モデル: 分析・リライト ／ 判断モデル: リライト案の品質検査")
            if st.session_state.model_management_supported:
                st.caption("未ロードの選択モデルは、処理開始時にLM Studioへ自動ロードします。")
            else:
                st.caption("このLM Studioでは自動ロードAPIを使えないため、ロード済みモデルのみ表示しています。")
            st.caption(f"LM Studioから {len(st.session_state.available_models)} 件取得")
        else:
            st.text_input("推論モデルID", key="model", help="一覧を取得できない場合はモデルIDを入力できます。")
            st.text_input(
                "判断モデルID（Jev）", key="decision_model_manual",
                on_change=_mark_decision_model_selected,
                help="品質検査に使うモデルID。空欄なら推論モデルを使います。",
            )
            if st.session_state.model_load_error:
                st.caption(f"モデル一覧を取得できません: {st.session_state.model_load_error}")
        st.selectbox("リライト強度", ["控えめ", "標準", "しっかり調整"], key="strength")
        st.toggle("履歴を保存する（初期状態: OFF）", key="save_history")
        st.toggle(
            "開発用ログを記録する（初期状態: OFF）", key="dev_logging",
            help="処理段階・所要時間・エラー種別・トークン数を記録します。入力文やモデル応答本文は記録しません。",
        )
        _render_development_log_controls(logger)
        if DEVELOPMENT_LOG_PATH.exists() and st.button("開発ログを消去"):
            logger.clear()
            st.rerun()
        st.caption("履歴OFF時は文章をアプリ独自の永続ストレージに保存しません。文章は推論モデルと判断モデルに送信されます。")
        if st.button("モデル一覧を更新"):
            try:
                models = _load_models(logger)
                st.success(f"モデル {len(models)} 件を取得しました")
            except Exception as exc:
                st.session_state.model_load_attempted = True
                st.session_state.model_load_error = type(exc).__name__
                logger.log("model_list_error", error_type=type(exc).__name__)
                st.error(f"LM Studioに接続できません: {type(exc).__name__}: {exc}")
        st.divider()
        _render_history(store)


def _render_risk_column(label: str, data: dict[str, Any]) -> None:
    st.subheader(label)
    scores = [risk["score"] for risk in data["risks"]]
    st.metric("リスクスコア（最大カテゴリ）", max(scores, default=0))
    st.caption(f"評価確信度: {data['confidence']} / 100 — 炎上確率や法的リスクではありません")
    if not data["risks"]:
        st.write("目立ったリスクは検出されませんでした。")
    for risk in data["risks"]:
        st.markdown(f"**{risk['category']} — {risk['score']}/100**")
        if risk["evidence"]:
            st.write(f"該当箇所: 「{risk['evidence']}」")
        st.write(risk["explanation"])
    st.caption(data["summary"])


def _render_result(result: dict[str, Any]) -> None:
    st.divider()
    st.subheader("リライト結果")
    st.text_area("コピーして使う文章", value=result["rewrite"], height=140, key="result_copy")
    st.caption("上の結果欄から文章を選択してコピーできます。")
    if not result["jev"]["passed"]:
        st.warning("品質検査で解消できなかった項目があります。内容を確認してから使用してください。")
        st.write("未解決:", result["jev"].get("unresolved", ""))
    with st.expander("原文（閲覧のみ）"):
        st.text(result["original"])

    before_column, after_column = st.columns(2)
    with before_column:
        _render_risk_column("リライト前", result["before"])
    with after_column:
        _render_risk_column("リライト後", result["after"])

    st.subheader("感情傾向（文章上の推定・0〜3）")
    st.bar_chart({"リライト前": result["before"]["emotions"],
                  "リライト後": result["after"]["emotions"]})
    st.caption("心理状態の実測ではありません。Jevの合否には使いません。")
    st.info(f"Jev: {'合格' if result['jev']['passed'] else '未解決'} · {result['jev']['attempts']}回検査")
    if result["jev"].get("issues"):
        st.write("検査メモ:", ", ".join(map(str, result["jev"]["issues"])))

    with st.form("revise_form"):
        followup = st.text_input("追加で修正する", placeholder="もっと柔らかく / この主張は残して")
        submitted = st.form_submit_button("追加指示で再リライト")
    if submitted and followup:
        try:
            with st.spinner("選択モデルを確認・ロードして追加修正中…"):
                updated = _run_pipeline(result["rewrite"], followup, _make_logger())
            updated["original"] = result["original"]
            st.session_state.result = updated
            if st.session_state.save_history:
                HistoryStore(HISTORY_DB_PATH).add(result["original"], updated)
            st.rerun()
        except Exception as exc:
            st.error(f"追加修正に失敗しました。現在の結果は保持されています: {type(exc).__name__}: {exc}")


def main() -> None:
    st.set_page_config(page_title="チャットリライター", page_icon="✍️", layout="wide")
    _initialize_state()
    logger = _make_logger()
    history_store = HistoryStore(HISTORY_DB_PATH)

    if (not st.session_state.model_load_attempted
            or st.session_state.model_management_supported is None):
        try:
            _load_models(logger)
        except Exception as exc:
            st.session_state.model_load_attempted = True
            st.session_state.model_load_error = type(exc).__name__
            logger.log("model_list_error", error_type=type(exc).__name__)

    st.title("✍️ チャットリライター")
    st.caption("送信前の短文を見直し、伝えたい主張を保ちながら表現を整えます。")
    _render_sidebar(logger, history_store)

    st.subheader("送信前の文章")
    source = st.text_area(
        "文章を入力してください", key="source", height=190, max_chars=5000,
        placeholder="例: 依頼した修正がまだ反映されていません。いつ対応できるか教えてください。",
    )
    character_count = len(source)
    st.caption(f"{character_count:,} / {MAX_INPUT_CHARACTERS:,} 文字（空白・改行を含む）")
    instruction = st.text_input("追加の修正指示（任意）", placeholder="例: もう少し簡潔に")
    if character_count > MAX_INPUT_CHARACTERS:
        st.error("4,999文字以内にしてください。長い文章は分割して入力してください。")

    if st.button("分析してリライト", type="primary",
                 disabled=not source.strip() or character_count > MAX_INPUT_CHARACTERS):
        try:
            with st.spinner("選択モデルを確認・ロードして分析・リライト中…"):
                result = _run_pipeline(source, instruction, logger)
            st.session_state.result = result
            st.session_state.error = None
            if st.session_state.save_history:
                history_store.add(source, result)
            st.rerun()
        except Exception as exc:
            logger.log("ui_processing_error", error_type=type(exc).__name__)
            st.session_state.error = f"処理に失敗しました ({type(exc).__name__}): {exc}"

    if st.session_state.error:
        st.error(st.session_state.error)
    if st.session_state.result:
        _render_result(st.session_state.result)
    _render_development_log_view(logger)
