# Docker環境のセットアップ

Kaggle公式のDockerイメージを使用してJupyter Notebook環境を構築する手順です。

## 📋 前提条件

- Docker Desktopがインストールされていること
- Apple Silicon(Mシリーズ)の場合は、Docker Desktopの設定でRosettaを有効化
  - Rosetta未導入の場合は以下を実行：
    ```bash
    softwareupdate --install-rosetta --agree-to-license
    ```

## 🚀 セットアップ手順

### 1. Kaggle公式イメージを取得（初回のみ）

```bash
docker pull gcr.io/kaggle-images/python
```

### 2. Notebook用コンテナを起動

```bash
docker run -it --rm \
  -p 8888:8888 \
  -v /Users/orimotoseiya/Desktop/Kaggle:/workspace \
  -v ~/.kaggle:/root/.kaggle \
  gcr.io/kaggle-images/python \
  bash
```

**オプション説明**:
- `-it`: インタラクティブモード
- `--rm`: コンテナ終了時に自動削除
- `-p 8888:8888`: ポートマッピング（ホスト:コンテナ）
- `-v /Users/orimotoseiya/Desktop/Kaggle:/workspace`: ワークスペースをマウント
- `-v ~/.kaggle:/root/.kaggle`: Kaggle APIトークンをマウント

成功するとプロンプトが `root@xxxx:/#` になります。

### 3. コンテナ内で作業ディレクトリへ移動

```bash
cd /workspace
```

**重要**: Jupyterを起動する前に必ず `/workspace` に移動してください。これにより、`notebooks/` ディレクトリのファイルが正しく表示されます。

### 4. Jupyter Notebookを起動

#### トークン認証あり（デフォルト、推奨）

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### トークン認証なし（個人使用のみ）

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

**オプション説明**:
- `--ip=0.0.0.0`: すべてのネットワークインターフェースでリッスン（Docker経由でアクセス可能）
- `--allow-root`: rootユーザーで実行するために必要
- `--NotebookApp.token=''`: トークン認証を無効化（個人使用のみ）
- `--NotebookApp.password=''`: パスワード認証を無効化

**注意**: トークンなしで起動した場合、`http://127.0.0.1:8888` で直接アクセスできます。**ローカル環境でのみ使用してください。**

**セキュリティ設定**:
- デフォルトでワンタイムトークンが発行されます
- 必要に応じて `jupyter notebook password` でパスワードを設定できます
- 外部公開する場合はSSHトンネルなどで保護してください
- `--NotebookApp.token=''` のような無防備設定は使わないでください

### 5. ブラウザでアクセス

1. `http://127.0.0.1:8888/tree` を開く
2. トークン（またはパスワード）でログイン
3. `notebooks/` ディレクトリで作業開始

## 🔧 よくある操作

### コンテナをバックグラウンドで起動

#### トークン認証あり

```bash
docker run -d \
  --name kaggle-notebook \
  -p 8888:8888 \
  -v /Users/orimotoseiya/Desktop/Kaggle:/workspace \
  -v ~/.kaggle:/root/.kaggle \
  -w /workspace \
  gcr.io/kaggle-images/python \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### トークン認証なし（個人使用のみ）

```bash
docker run -d \
  --name kaggle-notebook \
  -p 8888:8888 \
  -v /Users/orimotoseiya/Desktop/Kaggle:/workspace \
  -v ~/.kaggle:/root/.kaggle \
  -w /workspace \
  gcr.io/kaggle-images/python \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

**重要**: 
- `-w /workspace` オプションで作業ディレクトリを指定しています。これにより、Jupyterが `/workspace` で起動し、`notebooks/` ディレクトリのファイルが正しく表示されます。
- トークンなしで起動した場合、`http://127.0.0.1:8888` で直接アクセスできます。

### 実行中のコンテナに接続

```bash
docker exec -it kaggle-notebook bash
```

### コンテナを停止

```bash
docker stop kaggle-notebook
docker rm kaggle-notebook
```

## 📝 データパスの注意点

Docker環境では、以下のパス構造になります：

```
/workspace/              # マウントされたKaggleディレクトリ
├── notebooks/           # ノートブック
├── input/               # データ
│   └── titanic/
│       └── train.csv
└── submissions/        # 提出ファイル
```

ノートブック内では以下のようにパスを指定します：

```python
# 正しいパス
train_df = pd.read_csv('../input/titanic/train.csv')

# 間違ったパス（Kaggle環境用）
# train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
```

## 🐛 トラブルシューティング

### ポート8888が既に使用されている

別のポートを使用：
```bash
docker run -it --rm \
  -p 8889:8888 \
  ...
```

### 権限エラー

Docker Desktopの設定で、ファイル共有の権限を確認してください。

### 403 Forbidden エラー（ファイルが開けない）

**症状**: Jupyterにアクセスできるが、ファイルやディレクトリが開けない（403 Forbidden）

**原因**: Cookie認証の問題

**解決策**:
1. ブラウザのCookieをクリア（Jupyter関連のCookieを削除）
2. 新しいトークンでアクセス（コンテナを再起動して新しいトークンを取得）
3. シークレットモードでアクセスを試す

または、Jupyter Serverのバージョン問題の場合：
```bash
# コンテナ内で実行
pip install -U "jupyter-server<2.0.0"
# その後、Jupyterを再起動
```

### イメージが古い

最新のイメージを取得：
```bash
docker pull gcr.io/kaggle-images/python
```

### Optunaのインストール

Kaggle公式イメージにはOptunaが含まれていない場合があります。コンテナ内でインストールしてください：

```bash
# コンテナ内で実行
pip install optuna
```

または、コンテナ起動時にインストール：

```bash
docker run -it --rm \
  -p 8888:8888 \
  -v /Users/orimotoseiya/Desktop/Kaggle:/workspace \
  -v ~/.kaggle:/root/.kaggle \
  gcr.io/kaggle-images/python \
  bash -c "pip install optuna && bash"
```

**Optunaの確認方法**:
```bash
# コンテナ内で実行
python -c "import optuna; print(optuna.__version__)"
```

---

**関連ドキュメント**: [README.md](../README.md) | [Notebookの説明](notebooks.md)

