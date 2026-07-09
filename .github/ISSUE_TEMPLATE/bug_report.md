---
name: バグ報告
about: 動作しない・想定外の挙動を報告する
title: "[Bug] "
labels: bug
---

## 症状

<!-- 何が起きたか、期待していた動作は何か -->

## 再現手順

1.
2.
3.

## 環境

- OS:
- GPU: (例: RTX PRO 6000 Blackwell / なし)
- `IMAGE3D_GENERATOR`: (`mock` / `auto` / `hunyuan3d` / `pixal3d`)
- `GET /api/health` の出力:

```json

```

## エラーメッセージ・ログ

<!-- サーバのログ、ブラウザのコンソールエラーなど -->

```

```

## 確認事項

- [ ] `docs/SPEC.md` の README「よくある質問(FAQ)」を確認した
- [ ] `.venv/bin/pytest tests/ -q` は手元で通る
