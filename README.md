# SimpleLiveBlur

カメラ映像の顔や個人情報をリアルタイムでぼかし、OBSに配信するデスクトップアプリです。

## 特徴

- **リアルタイム顔ブラー** — カメラ映像から顔を検出し自動でぼかす
- **複数のプライバシー対象** — 顔・人物・画面（TV/PC/スマホ）・ナンバープレートに対応
- **GUIで簡単操作** — ブラー対象のON/OFF、カメラ切り替え、強度調整をワンクリック
- **OBS連携** — MJPEGストリームとして配信し、OBSのメディアソースで受信
- **クロスプラットフォーム** — macOS / Windows 対応

## ダウンロード

[Releases](../../releases) ページから最新版をダウンロードしてください。

| OS | ファイル |
|----|---------|
| macOS | `SimpleLiveBlur-macOS.zip` |
| Windows | `SimpleLiveBlur-Windows.zip` |

### macOS での初回起動

署名されていないアプリのため、初回はGatekeeperにブロックされます。以下のいずれかの方法で開いてください。

**方法1：右クリックで開く**

アプリを右クリック（またはControl+クリック）→「開く」→ 確認ダイアログで「開く」をクリック

**方法2：ターミナルで属性を解除**

ZIPを展開したフォルダで以下を実行してください。

```bash
xattr -cr SimpleLiveBlur.app
```

※ 必要に応じて `SimpleLiveBlur.app` を `/Applications` にドラッグしてインストールできます。一度開けば、以降は通常通りダブルクリックで起動できます。

### Windows での初回起動

署名されていないアプリのため、SmartScreenの警告が表示されます。

「詳細情報」をクリック →「実行」をクリックで起動できます。

## ソースからのセットアップ

### 必要なもの

- Python 3.9 以上
- Webカメラ

### インストール

```bash
git clone https://github.com/YOUR_USERNAME/SimpleLiveBlur.git
cd SimpleLiveBlur

pip install -r requirements.txt
```

### モデルファイルのダウンロード

```bash
# 顔検出モデル
curl -o blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

# 物体検出モデル（人物・画面検出用）
curl -o efficientdet_lite0.tflite \
  https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite
```

### 起動

```bash
python face_blur_stream.py
```

## 使い方

### 1. アプリを起動する

起動するとGUIウィンドウが開き、カメラのプレビューが表示されます。

### 2. ブラー対象を選ぶ

チェックボックスでぼかす対象を選択します。

| 対象 | 説明 | デフォルト |
|------|------|-----------|
| 顔 | 人の顔を検出してぼかす | ON |
| 人物 | 人物全体をぼかす | OFF |
| 画面 | TV・PC・スマホの画面をぼかす | OFF |
| ナンバー | 車のナンバープレートをぼかす | OFF |

### 3. OBSに接続する

1. OBSでソース「+」→「メディアソース」を追加
2. 「ローカルファイルを使用」のチェックを外す
3. 入力URLに `http://localhost:8080/` を入力
4. OKを押す

これでブラー処理済みの映像がOBSに表示されます。

### 4. カメラを切り替える

GUIの「カメラ」ドロップダウンから別のカメラを選択できます。配信を止めずに切り替わります。

### 5. 詳細設定を調整する（任意）

「詳細設定」をクリックするとスライダーが表示されます。

| 設定 | 説明 | デフォルト |
|------|------|-----------|
| ブラー強度 | ぼかしの強さ（大きいほど強い） | 99 |
| ブラー拡散 | ぼかしの広がり | 30 |
| 顔検出閾値 | 顔検出の感度（低いほど検出しやすい） | 0.50 |
| 物体検出閾値 | 人物・画面検出の感度 | 0.40 |

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| カメラが映らない | ドロップダウンで別のカメラに切り替え |
| 顔を検出しない | 詳細設定で顔検出閾値を下げる / 照明を改善 |
| OBSで映像が出ない | アプリ起動後にOBSのメディアソースを再接続 |
| 処理が重い | 不要なブラー対象をOFFにする |

## ビルド

### macOS

```bash
pyinstaller SimpleLiveBlur.spec
# dist/SimpleLiveBlur.app が生成される
```

### Windows

```bash
pyinstaller SimpleLiveBlur_win.spec
# dist\SimpleLiveBlur\ が生成される
```

## ライセンス

MIT
