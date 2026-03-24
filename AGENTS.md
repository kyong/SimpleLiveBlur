# AGENTS.md - faceBlur プロジェクト

## プロジェクト概要

OBSでリアルタイム顔ブラー配信を行うPythonスクリプト。
カメラ映像を取得し、MediaPipeで顔検出 → GaussianBlurでぼかし → MJPEGストリームとして
`http://localhost:8080/` に配信。OBSのメディアソースで受け取り配信する。

---

## 構成

```
物理カメラ
  └─ Python（face_blur_stream.py）
       ├─ MediaPipe Tasks API で顔検出
       ├─ OpenCV GaussianBlur でぼかし
       ├─ PyQt6 GUI（プレビュー表示・カメラ切り替え）
       └─ MJPEGサーバー（localhost:8080）
             └─ OBS メディアソース → 配信
```

### スレッド構成

| スレッド           | 役割                                  |
|--------------------|---------------------------------------|
| メインスレッド      | PyQt6 GUI（イベントループ）            |
| デーモンスレッド 1  | カメラキャプチャ + 顔検出 + ブラー処理 |
| デーモンスレッド 2  | MJPEGサーバー（HTTP配信）              |

---

## ファイル構成

```
faceBlur/
├── face_blur_stream.py          # メインスクリプト
├── blaze_face_short_range.tflite  # MediaPipeモデルファイル（要ダウンロード）
└── AGENTS.md                    # このファイル
```

---

## 環境

| 項目 | 内容 |
|------|------|
| OS | macOS（Apple Silicon / ARM） |
| OBS | 32.1.0 |
| Python | 3.x |
| mediapipe | 0.10.33（Tasks API使用） |
| opencv-python | 4.13.0.92 |
| PyQt6 | 6.10.x（GUI） |

---

## 依存パッケージ

```bash
pip install opencv-python mediapipe PyQt6
```

※ `pyvirtualcam` はmacOS ARMに非対応のため不使用。  
※ OBSプラグイン `obs-backgroundremoval` はmacOSのGatekeeperにより読み込み失敗のため不使用。

---

## モデルファイルのダウンロード

```bash
curl -o blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```

スクリプトと同じディレクトリに配置すること。

---

## 起動手順

```bash
# スクリプトのあるディレクトリで実行
python face_blur_stream.py
```

起動するとGUIウィンドウが表示される。
プレビューでブラー処理後の映像を確認でき、ドロップダウンでカメラを切り替え可能。

OBSで以下を設定：

```
ソース「+」→「メディアソース」
→「ローカルファイルを使用」のチェックを外す
→ 入力URL: http://localhost:8080/
→ OK
```

---

## 主要パラメータ（face_blur_stream.py）

| パラメータ | 場所 | デフォルト値 | 説明 |
|-----------|------|-------------|------|
| `padding` | `blur_faces()` | `0.25` | 顔領域の余白倍率。大きいほど広くぼかす |
| `(99, 99)` | `GaussianBlur` | — | ブラーカーネルサイズ。大きいほど強くぼかす |
| `30` | `GaussianBlur` | — | ブラーの標準偏差。大きいほど強い |
| `model_asset_path` | `BaseOptions` | `blaze_face_short_range.tflite` | モデルファイルパス |
| `min_detection_confidence` | `FaceDetectorOptions` | `0.5` | 顔検出の閾値（0.0〜1.0） |
| `cv2.VideoCapture(0)` | `capture_loop()` | `0` | カメラデバイス番号 |
| `IMWRITE_JPEG_QUALITY` | `imencode` | `85` | MJPEG品質（1〜100） |
| `port` | `__main__` | `8080` | MJPEGサーバーのポート番号 |

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| カメラが映らない | GUIのドロップダウンで別のカメラに切り替え |
| 顔を検出しない | `min_detection_confidence` を下げる / 照明を改善 |
| 横顔・遠距離の顔を検出しない | モデルを `blaze_face_full_range.tflite` に変更（要別途ダウンロード） |
| OBSで映像が出ない | スクリプト起動後にOBSのメディアソースを再接続 |
| 処理が重い | `IMWRITE_JPEG_QUALITY` を下げる / 解像度を落とす |
| ポート競合エラー | `port = 8080` を別番号（例: `8081`）に変更 |

---

## 既知の制限

- 横顔・マスク着用・逆光など条件が悪い場合は検出漏れあり
- MJPEGストリーム経由のため数フレーム（~100ms）の遅延が発生する
- `blaze_face_short_range` は近距離向け。遠距離の場合は `full_range` モデルを使用

---

## 今後の改善候補

- [ ] 顔以外の個人特定情報（テキスト・ナンバープレート等）のマスク対応
- [x] GUIでカメラ切り替え・プレビュー表示
- [ ] GUIで閾値・ブラー強度をリアルタイム調整
- [ ] PyInstallerによるインストーラー作成（Win/Mac対応）
- [ ] 検出漏れ対策として複数フレームの検出結果をマージ
- [ ] WebSocket経由での制御（開始・停止・パラメータ変更）