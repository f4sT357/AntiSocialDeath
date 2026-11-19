# チャットリライター

短文を送る前に、ローカルのLM Studioで文章を分析し、主張を保ちながらリライトするデスクトップ向けアプリです。

## 起動

Python 3.10以上が必要です。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

LM Studioのローカルサーバーを起動してください。既定の接続先は `http://localhost:1234/v1` です。アプリ起動時にLM Studioから利用可能なLLMを取得し、未ロードのモデルも一覧に表示します。設定サイドバーの「推論モデル」は分析・リライトに、「判断モデル（Jev）」はリライト案の品質検査に使います。初回は利用可能なモデル一覧からLFM2.5を判断モデルに自動選択し、選択後は個別に変更できます。「推論モデルと同じ」を選ぶと従来どおり1モデルで動きます。選択モデルが未ロードなら、文章の処理開始時にLM StudioのREST APIで自動ロードします。「モデル一覧を更新」で一覧とロード状態を再取得できます。LM Studio 0.4以降のモデル管理APIがない場合は、ロード済みモデルのみ選択できます。履歴保存は既定でOFFです。

リライト品質検査はLangGraphの循環ワークフローで実行します。リライト案を判断モデルが検査し、不合格なら指摘を推論モデルへ渡して再生成します。合格で終了し、合格しない場合も `MAX_JEV_REWRITE_ATTEMPTS`（既定3回）で停止して未解決として表示します。

開発時はサイドバーの「開発用ログを記録する」をONにすると、処理段階・所要時間・エラー種別・トークン数を `data/development.jsonl` に記録します。入力文、リライト結果、モデル応答本文は記録しません。サイドバーからダウンロードまたは消去できます。既定はOFFです。

## コード構成

- `app.py`: Streamlitの起動点
- `chat_rewriter/ui.py`: 画面とユーザー操作
- `chat_rewriter/service.py`: 分析・リライト・Jevをつなぐ処理フローとLangGraphの上限付き検査ループ
- `chat_rewriter/llm.py`: LM Studio接続、JSON修復、応答形式検証
- `chat_rewriter/model_manager.py`: LM Studioのローカルモデル一覧取得と自動ロード
- `chat_rewriter/storage.py`: SQLite履歴とCSV/JSON出力
- `chat_rewriter/diagnostics.py`: 開発ログ
- `chat_rewriter/config.py`: パス、既定値、カテゴリ

## 注意

リスク・感情スコアは文章からの推定で、炎上確率、心理状態、法律判断を示しません。入力は設定したLM Studioサーバーに送信されます。接続先を確認してから利用してください。
