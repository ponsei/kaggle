# Titanic Starter Project

初心者向けにKaggleのTitanicコンペを学ぶためのプロジェクトです。

## ディレクトリ構成
- `notebooks/`: EDAや特徴量作成、モデリングを行うJupyter Notebookを配置
- `src/`: 再利用可能なPythonスクリプトや補助関数を配置
- `input/`: Kaggleからダウンロードした生データ(zip含む)を配置(Git追跡しない)
- `data/`: 前処理済みデータや中間生成物。必要なら追加

## これからの流れ
1. Kaggle APIトークンを `~/.kaggle/kaggle.json` に配置し、アクセス権を`600`に設定
2. `kaggle competitions download -c titanic -p input/` でデータ取得し、zipを展開
3. `notebooks/` にEDA兼モデリングNotebookを作成
4. モデルを評価し、`test.csv`から`submission.csv`を生成
5. `kaggle competitions submit -c titanic -f submission.csv -m "First submission"` で提出

## Jupyter Notebook（Docker経由）の起動手順

> **メモ**: Apple Silicon(Mシリーズ)ではDocker Desktopの設定でRosettaを有効化しておく。Rosetta未導入なら `softwareupdate --install-rosetta --agree-to-license` を先に実行。

1. Kaggle公式イメージを取得（初回のみ）
   ```bash
   docker pull gcr.io/kaggle-images/python
   ```
2. Notebook用コンテナを起動（改行ごとにEnterを押して入力する）
   ```bash
   docker run -it --rm \
     -p 8888:8888 \
     -v /Users/orimotoseiya/Desktop/Kaggle:/workspace \
     -v ~/.kaggle:/root/.kaggle \
     gcr.io/kaggle-images/python \
     bash
   ```
   - 成功するとプロンプトが `root@xxxx:/#` になる。
3. コンテナ内で作業ディレクトリへ移動
   ```bash
   cd /workspace
   ```
4. Jupyter Notebook（Classic UI）をトークン無しで起動
   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser \
     --NotebookApp.token='' --NotebookApp.password=''
   ```
5. ブラウザで `http://127.0.0.1:8888/tree` を開く  
   - ログイン不要でNotebook一覧が表示される。
   - `notebooks/` に `titanic-starter.ipynb` を作成し、以下の順で進める。
5. Notebookの進め方メモ
   1. `train.csv` の読み込みと欠損確認
   2. 基本統計・カテゴリ分布のチェック
   3. `SimpleImputer` + One-Hot Encodingで前処理
   4. ロジスティック回帰→ランダムフォレストでベースライン
   5. 精度確認→`test.csv` 推論→`submission.csv` 生成
# titanic_kaggle
# titanic_kaggle
