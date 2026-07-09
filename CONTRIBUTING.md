# コントリビューションガイド

バグ報告・機能要望はIssue、修正・機能追加はPull Requestで受け付けています。

## Issue

テンプレート(バグ報告 / 機能要望 / 質問)から作成してください。
セットアップ関連の質問は、まず [README.md](README.md) の
「よくある質問(FAQ)」をご確認ください。

## Pull Request

1. `master` からブランチを切る
2. 変更後、必ず以下を実行して green を確認する:
   ```bash
   .venv/bin/pytest tests/ -q
   ```
   (`.venv` が無い場合は README「セットアップ」参照。GPU/実モデルに触れない
   変更であれば `IMAGE3D_GENERATOR=mock` のままでテストは完結する)
3. UIに関わる変更は、実際にブラウザで動作確認する
4. PRテンプレートのチェックリストに従って提出する

## 設計を変更する場合

仕様変更・機能追加は [docs/SPEC.md](docs/SPEC.md)(要件)・
[docs/DEVELOPMENT_POLICY.md](docs/DEVELOPMENT_POLICY.md)(技術方針)・
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)(実装計画)を
先に更新してから着手すると、既存の設計判断(座標系、ジェネレータの
プラガブル設計など)との整合が取りやすくなります。

## ライセンスに関する注意

このリポジトリ自体は [Polyform Small Business License](LICENSE) です。
GPU実モデル(Hunyuan3D-2 / Pixal3D)は別ライセンスの外部リポジトリを
`third_party/` に別途導入する構成のため、それらのコードはこのリポジトリに
含まれません(詳細はREADME「ライセンス」節参照)。
