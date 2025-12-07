# RunPod NeuroQ エンドポイント - トラブルシューティングガイド

## 🔴 現在の問題

### 状況

1. **エンドポイントステータス**: Initializing
2. **ワーカーの状態**:
   - ✅ 1つのワーカーが初期化中（バージョン33 - Latest）
   - ❌ 4つのワーカーがunhealthy（バージョン30-32 - Outdated）
3. **ロールアウト進捗**: 1/5 (20%)のみが最新設定
4. **ジョブ**: 1件がキューで待機中

### エラーログで確認された問題

Dockerイメージプル時に以下のエラーが5回発生：

```
failed to pull image: failed to register layer: invalid output path: 
stat /var/lib/docker/165536.165536/overlay2/...: no such file or directory
```

**大きなレイヤー（2.489GB）のダウンロード後にレイヤー登録に失敗**

## 🎯 解決策

### ステップ1: unhealthyワーカーの削除（推奨）

RunPodダッシュボードで以下を実行：

1. **Workersタブ**を開く
2. 各unhealthyワーカーの**ゴミ箱アイコン**をクリックして削除
   - `qsoqrsoautsdfq` (バージョン30)
   - `sxe24taw5mabrp` (バージョン31)
   - `r5765xrx2secvd` (バージョン31)
   - `oc1fblu8642gou` (バージョン32)

これにより、新しいワーカーが最新バージョンで自動的に作成されます。

### ステップ2: エンドポイントの再デプロイ

unhealthyワーカーを削除しても問題が解決しない場合：

1. RunPodダッシュボードで「**Manage**」ドロップダウンを開く
2. 「**Update**」または「**Redeploy**」を選択
3. 最新のDockerイメージで再デプロイ

### ステップ3: 最新ワーカーのログ確認

初期化中のワーカー（`329q6uduqaos2z`）のログを確認：

1. **Workersタブ**でワーカーをクリック
2. **Logsタブ**で詳細ログを確認
3. エラーが発生していないか確認

### ステップ4: RunPodサポートへの連絡

問題が続く場合：

1. 以下の情報を準備：
   - エンドポイントID: `p32ooi7hdvi1jo`
   - エラーログ: `/Users/yuyahiguchi/Downloads/logs.txt`
   - ワーカーIDとステータス
   - 発生時刻

2. RunPodサポートに連絡：
   - ダッシュボードの「Help & resources」からサポートにアクセス
   - または、support@runpod.io にメール

## 📋 ワーカー一覧

| Worker ID | ステータス | バージョン | アクション |
|-----------|-----------|-----------|-----------|
| `329q6uduqaos2z` | `initializing` | 33 (Latest) | ⏳ 監視（正常） |
| `qsoqrsoautsdfq` | `× unhealthy` | 30 (Outdated) | 🗑️ 削除推奨 |
| `sxe24taw5mabrp` | `× unhealthy` | 31 (Outdated) | 🗑️ 削除推奨 |
| `r5765xrx2secvd` | `Extra, × unhealthy` | 31 (Outdated) | 🗑️ 削除推奨 |
| `oc1fblu8642gou` | `Extra, × unhealthy` | 32 (Outdated) | 🗑️ 削除推奨 |

## 🔍 根本原因の分析

### Dockerイメージの問題

1. **大きなレイヤーサイズ**: 2.489GBのレイヤーが問題
2. **ストレージドライバーの問題**: overlay2でのレイヤー登録失敗
3. **RunPodサーバー側の問題**: ディスク容量やストレージ設定の問題

### 対処方法

#### 短期的な解決策

1. ✅ unhealthyワーカーを削除
2. ✅ エンドポイントを再デプロイ
3. ✅ RunPodサポートに連絡

#### 長期的な解決策

1. **Dockerfileの最適化**:
   - より軽量なベースイメージを使用
   - マルチステージビルドの最適化
   - 不要なファイルの削除

2. **イメージサイズの削減**:
   - 不要な依存関係の削除
   - キャッシュの最適化
   - .dockerignoreファイルの使用

## 🚀 次のアクション

### 今すぐ実行

1. ✅ RunPodダッシュボードを開く
2. ✅ Workersタブでunhealthyワーカーを確認
3. ✅ 各unhealthyワーカーを削除
4. ✅ 新しいワーカーの作成を待つ

### 5分後

5. ✅ 最新ワーカー（`329q6uduqaos2z`）のログを確認
6. ✅ 初期化が完了しているか確認
7. ✅ ジョブが正常に実行されるか確認

### 問題が続く場合

8. ✅ エンドポイントを再デプロイ
9. ✅ RunPodサポートに連絡

## 📊 監視ポイント

以下を定期的に確認：

- **ワーカーのステータス**: unhealthy状態が解消されたか
- **ロールアウト進捗**: 5/5 (100%)になっているか
- **ジョブの実行**: キュー待ちのジョブが正常に処理されるか
- **コスト**: 正常に動作している場合、コストが発生する

## 関連ドキュメント

- ログ分析: `LOGS_ANALYSIS.md`
- ワーカーステータス分析: `WORKER_STATUS_ANALYSIS.md`
- Dockerfile: `Dockerfile`

