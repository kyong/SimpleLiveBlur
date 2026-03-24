# AGENTS.md - SimpleLiveBlur

## プロジェクト概要

OBSでリアルタイム個人情報ブラー配信を行うPythonアプリケーション。
カメラ映像を取得し、顔・人物・画面・ナンバープレートを検出 → GaussianBlurでぼかし →
MJPEGストリームとして `http://localhost:8080/` に配信。OBSのメディアソースで受け取り配信する。
GUIからブラー対象のON/OFFやブラー強度・検出閾値をリアルタイムで調整可能。

---

## 構成

```
物理カメラ
  └─ Python（face_blur_stream.py）
       ├─ MediaPipe FaceDetector で顔検出
       ├─ MediaPipe ObjectDetector で人物・画面検出
       ├─ OpenCV Haar Cascade でナンバープレート検出
       ├─ OpenCV GaussianBlur でぼかし
       ├─ PyQt6 GUI（プレビュー表示・カメラ切り替え）
       └─ MJPEGサーバー（localhost:8080）
             └─ OBS メディアソース → 配信
```

### スレッド構成

| スレッド           | 役割                                  |
|--------------------|---------------------------------------|
| メインスレッド      | PyQt6 GUI（イベントループ）            |
| デーモンスレッド 1  | カメラキャプチャ + PII検出 + ブラー処理 |
| デーモンスレッド 2  | MJPEGサーバー（HTTP配信）              |

---

## ファイル構成

```
SimpleLiveBlur/
├── face_blur_stream.py            # メインスクリプト
├── blaze_face_short_range.tflite  # 顔検出モデル（要ダウンロード）
├── efficientdet_lite0.tflite      # 物体検出モデル（要ダウンロード）
├── requirements.txt               # 依存パッケージ
├── SimpleLiveBlur.spec            # PyInstaller設定（macOS）
├── SimpleLiveBlur_win.spec        # PyInstaller設定（Windows）
├── .github/workflows/build.yml    # GitHub Actions CI/CD
└── AGENTS.md                      # このファイル
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
| PyInstaller | 6.x（ビルド・配布） |

---

## 依存パッケージ

```bash
pip install -r requirements.txt
```

※ `pyvirtualcam` はmacOS ARMに非対応のため不使用。  
※ OBSプラグイン `obs-backgroundremoval` はmacOSのGatekeeperにより読み込み失敗のため不使用。

---

## モデルファイルのダウンロード

```bash
# 顔検出モデル
curl -o blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

# 物体検出モデル（人物・画面検出用）
curl -o efficientdet_lite0.tflite \
  https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite
```

スクリプトと同じディレクトリに配置すること。
ナンバープレート検出はOpenCV内蔵のHaar Cascadeを使用するため、追加ダウンロード不要。

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

| パラメータ | デフォルト値 | GUI調整 | 説明 |
|-----------|-------------|---------|------|
| `blur_strength` | `99` | スライダー（1〜199） | ブラーカーネルサイズ。大きいほど強くぼかす |
| `blur_sigma` | `30` | スライダー（1〜100） | ブラーの標準偏差。大きいほど拡散する |
| `face_threshold` | `0.50` | スライダー（0.01〜1.00） | 顔検出の閾値。低いほど検出しやすい |
| `object_threshold` | `0.40` | スライダー（0.01〜1.00） | 物体検出の閾値。低いほど検出しやすい |
| `padding` | `0.25` | — | 顔領域の余白倍率 |
| `IMWRITE_JPEG_QUALITY` | `85` | — | MJPEG品質（1〜100） |
| `port` | `8080` | — | MJPEGサーバーのポート番号 |

※ GUIの「詳細設定」を開くとスライダーが表示される（デフォルトは折りたたみ）。

### ブラー対象（GUIチェックボックスで切り替え）

| 対象 | 検出方法 | デフォルト |
|------|---------|-----------|
| 顔 | MediaPipe FaceDetector | ON |
| 人物 | MediaPipe ObjectDetector（COCO "person"） | OFF |
| 画面 | MediaPipe ObjectDetector（COCO "tv", "laptop", "cell phone"） | OFF |
| ナンバープレート | OpenCV Haar Cascade | OFF |

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| カメラが映らない | GUIのドロップダウンで別のカメラに切り替え |
| 顔を検出しない | GUIの詳細設定で顔検出閾値を下げる / 照明を改善 |
| 横顔・遠距離の顔を検出しない | モデルを `blaze_face_full_range.tflite` に変更（要別途ダウンロード） |
| OBSで映像が出ない | スクリプト起動後にOBSのメディアソースを再接続 |
| 処理が重い | 不要なブラー対象をOFFにする / `IMWRITE_JPEG_QUALITY` を下げる / 解像度を落とす |
| ポート競合エラー | `port = 8080` を別番号（例: `8081`）に変更 |

---

## ビルド・配布

### ローカルビルド（macOS）

```bash
pip install -r requirements.txt
pyinstaller SimpleLiveBlur.spec
# dist/SimpleLiveBlur.app が生成される
```

### ローカルビルド（Windows）

```bash
pip install -r requirements.txt
pyinstaller SimpleLiveBlur_win.spec
# dist\SimpleLiveBlur\ フォルダが生成される
```

### GitHub Actionsによる自動ビルド

タグをpushするとWin/Mac向けにビルドされ、GitHub Releasesに自動添付される。

```bash
git tag v1.0.0
git push origin v1.0.0
```

Release添付物:
- `SimpleLiveBlur-macOS.zip` — macOS .app バンドル
- `SimpleLiveBlur-Windows.zip` — Windows実行ファイル

---

## 既知の制限

- 横顔・マスク着用・逆光など条件が悪い場合は検出漏れあり
- MJPEGストリーム経由のため数フレーム（~100ms）の遅延が発生する
- `blaze_face_short_range` は近距離向け。遠距離の場合は `full_range` モデルを使用

---

## 今後の改善候補

- [x] 顔以外の個人特定情報（人物・画面・ナンバープレート）のマスク対応
- [x] GUIでカメラ切り替え・プレビュー表示
- [x] GUIでブラー対象のON/OFF切り替え
- [x] GUIで閾値・ブラー強度をリアルタイム調整（詳細設定パネル）
- [x] PyInstallerによるインストーラー作成（Win/Mac対応）
- [x] GitHub Actions CI/CDでリリース自動化
- [ ] 検出漏れ対策として複数フレームの検出結果をマージ
- [ ] WebSocket経由での制御（開始・停止・パラメータ変更）