# cube_projection

K-POPなどのミュージックビデオからメンバーごとのカットを自動抽出し、6人分の映像を立方体の各面に貼り付けてリアルタイムに回転表示するデモです。姿勢センサー（マイコン）からの姿勢角（Roll/Pitch/Yaw）で視点を操作し、音声は同期再生します。

## 概要
- 顔認識: InsightFace を用いて学習（参照画像から埋め込みを作成）。
- カット生成: `mvcreator.py` が MV からフレームを走査 → 同一人物を追跡 → 代表人物名を決定 → 顔中心にズームしたメンバー別動画を書き出し。
- 投影表示: `cube_mapping.py` が 6 本の動画を立方体の各面に貼り付け、シリアルから受け取る R/P/Y でカメラ姿勢を更新。音声（mp3）をループ再生。

## リポジトリ構成（抜粋）
```
cube_mapping.py      # 立方体へ動画テクスチャを貼って表示（OpenGL）
mvcreator.py         # 顔認識→人物追跡→メンバー別カット動画の生成
bts/                 # 学習用の参照画像（人物ごとにサブフォルダ）
bts_bin/             # 顔埋め込み（encodings.npy）と名前（names.npy）の保存先
newjeans/
newjeans_bin/
twice/
twice_bin/
```

各グループのフォルダ配下は「人物名/画像.jpg...」という構造を想定しています（例: `twice/nayeon/xxx.jpg`）。`*_bin/` 配下には生成済みの顔埋め込みが保存されます。

## 必要環境
- macOS（動作確認の想定）
- Python 3.9+ くらいを推奨
- カメラ姿勢入力用のマイコン（任意。無い場合はコード内で姿勢計算の箇所を無効化/固定しても可）

### 主要パッケージ
- numpy
- opencv-python
- insightface（+ onnxruntime などのバックエンド）
- PyOpenGL, PyOpenGL_accelerate
- pygame（音声再生）
- pyserial（マイコンとシリアル接続）

> 注: InsightFace は GPU が無い環境では `ctx_id=-1`（CPU）で動かすのが安定です（後述）。macOS Apple Silicon では `onnxruntime` か `onnxruntime-silicon` を選択します。

## セットアップ
仮想環境の作成と依存インストール（例）:

```bash
# 任意: 仮想環境
python3 -m venv .venv
source .venv/bin/activate

# パッケージ（最小例）
pip install numpy opencv-python insightface onnxruntime PyOpenGL PyOpenGL_accelerate pygame pyserial
```

GPU/CPU によって InsightFace のバックエンドは変わります。Apple Silicon では以下のような組み合わせもあります。

```bash
# Apple Silicon の場合の一例（必要に応じて）
pip install onnxruntime-silicon
```

## 顔埋め込み（モデル）の準備
`mvcreator.py` の `Model` クラスは、既に `*_bin/encodings.npy` があれば読み込み、無ければ生成します。

- 参照画像の配置例（Twice の場合）
  - `twice/nayeon/xxx.jpg`, `twice/jihyo/yyy.jpg`, ...
- 生成物の保存先（デフォルト）
  - `twice_bin/encodings.npy`
  - `twice_bin/names.npy`

必要に応じて以下を変更してください（`mvcreator.py` 内）：
- `Model.binary_path` … 出力先（例: `twice_bin`, `bts_bin`, `newjeans_bin`）
- `Model.create_model()` の `people_image_dir_path` … 参照画像のパス（例: `twice`, `bts`, `newjeans`）

InsightFace の初期化箇所（スクリプト末尾）：
```python
analyzer = FaceAnalysis()
# GPU なしの Mac などでは CPU 実行にする
analyzer.prepare(ctx_id=-1)
```

## メンバー別カット動画の生成（mvcreator.py）
`mvcreator.py` は以下の流れで動画を書き出します。
1. MV（入力動画）を読み込み
2. 各フレームで顔検出＆埋め込み抽出
3. 顔の移動距離から同一人物のシーケンスを構築
4. シーケンス全体で最頻の人物名を採用
5. 顔中心にズームしたフレームを生成し、人物ごとの動画として出力

デフォルトでは以下の設定になっています（必要に応じて変更してください）。
- 入力動画: `heartshaker.mp4`
- 対象フレーム範囲: 1000〜1200 フレーム
- 出力先: 実行時刻のディレクトリ（例: `20250101123045/`）に「人物名.mp4」を保存

実行例:
```bash
python mvcreator.py
```

変更ポイント（例）:
- 別の動画にする: `movie = Movie('your_video.mp4', start, step, end)` を編集
- グループを変える: `Model.binary_path` と `people_image_dir_path` を `bts`/`newjeans` に変更
- CPU 実行にする: `analyzer.prepare(ctx_id=-1)` に変更

## 立方体への投影表示（cube_mapping.py）
`cube_mapping.py` は 6 面に 6 本の動画を貼ります。タイトルにより、各面に対応するファイル名（人物名）が決まります。

スクリプト内の `names_dict` 例:
```python
names_dict = {
    "heartshaker": ["chaeyoung", "jihyo", "nayeon", "tzuyu", "dahyun", "momo"],
    "dynamite": ["jhope", "jungkook", "jin", "v", "jimin", "rm"],
}
```

準備するもの:
- 音声: `heartshaker.mp3`（タイトル名と同名）
- 動画フォルダ: `heartshaker/` 配下に `chaeyoung.mp4`, `jihyo.mp4`, ... の 6 本
- マイコン: シリアルで Roll/Pitch/Yaw を送信（`roll,pitch,yaw\n`）
  - macOS のデバイス名は `/dev/tty.usbserial-XXXX` 等になることが多いです。`cube_mapping.py` の `serial.Serial('/dev/ttyUSB0', 9600)` を環境に合わせて変更してください。

実行例:
```bash
python cube_mapping.py heartshaker
```

ウィンドウが開き、マイコンの姿勢に連動して立方体が回転します。動画はフレーム進行を補正しつつループ再生、音声はループ再生します。

> メンバー名の数は必ず 6 名分にしてください（立方体の 6 面に対応）。`mvcreator.py` の出力を揃えるか、`names_dict` と動画ファイル一式を整合させてください。

## トラブルシューティング
- シリアル接続が開けない: mac のポート名を確認し、権限（`/dev/tty.*`）を付与。不要なら `display()` 内のシリアル読み取りを一時的に無効化可。
- 動画が見つからない/真っ黒: フォルダ名・ファイル名が `names_dict` と一致しているか、拡張子・相対パスを確認。
- InsightFace が動かない: `onnxruntime` の種類（CPU/GPU/Apple Silicon）と `ctx_id` を見直し。
- OpenGL エラー: macOS ではシステム OpenGL/GLUT を利用。仮想環境の PyOpenGL を入れ直すと改善する場合があります。

## ライセンス
本リポジトリ内のスクリプトは研究・学習目的を想定しています。使用する動画・画像・音声の著作権には十分ご注意ください。

## 謝辞
- [InsightFace](https://github.com/deepinsight/insightface)
- OpenCV / PyOpenGL / pygame
