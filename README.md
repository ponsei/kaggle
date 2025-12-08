# Kaggle Competition Learning Project

初心者向けにKaggleコンペティションを学ぶためのプロジェクトです。複数のコンペに取り組み、機械学習の基礎から応用までを学習します。

## 📁 ディレクトリ構成

```
Kaggle/
├── competitions/      # 各コンペティションごとのディレクトリ
│   ├── titanic/
│   │   ├── configs/           # 設定ファイル（default.json）
│   │   ├── data/
│   │   │   ├── input/         # オリジナルの生データ
│   │   │   └── output/        # 処理されたデータや予測結果
│   │   ├── features/          # 特徴量エンジニアリング関連ファイル
│   │   ├── logs/              # 実行ログやモデルの学習ログ
│   │   ├── notebooks/         # Jupyter Notebook（EDA、特徴量作成、モデリング）
│   │   └── submissions/       # 提出ファイル（CSV）
│   ├── bnp-paribas/
│   ├── house-prices/
│   ├── atmacup8/
│   └── playground-series-s5e12/
├── src/               # 再利用可能なPythonスクリプト、補助関数、Tips系ドキュメント
```

## 🎯 取り組んでいるコンペティション

### 1. **Titanic - Machine Learning from Disaster** (分類)
- **タスク**: 二値分類（生存予測）
- **評価指標**: Accuracy
- **ノートブック**: 
  - `Titanic Kaggle.ipynb` - 基本実装
  - `Titanic_LightGBM.ipynb` - LightGBMを使用した実装
  - `Titanic Top Solution (Clean Version).ipynb` - 上位解法の実装

### 2. **BNP Paribas Cardif Claims Management** (分類)
- **タスク**: 二値分類（保険請求の管理）
- **評価指標**: Log Loss（対数損失）
- **ノートブック**: 
  - `BNP_Paribas_Cardif_Starter.ipynb` - 初心者向け解説付き

### 3. **House Prices: Advanced Regression Techniques** (回帰)
- **タスク**: 回帰（住宅価格予測）
- **評価指標**: RMSE（Root Mean Squared Error）
- **ノートブック**: 
  - `House_Prices_Starter.ipynb` - 初心者向け解説付き
  - `House_Prices_Comprehensive_EDA.ipynb` - 包括的なEDA

### 4. **atmaCup#8** (その他)
- **ノートブック**: `atmaCup#8.ipynb`

### 5. **Playground Series S5E12 - Diabetes Prediction Challenge** (分類)
- **タスク**: 二値分類（糖尿病診断予測）
- **評価指標**: Log Loss（対数損失）
- **ノートブック**: `Playground Series S5E12.ipynb`

## 📚 ドキュメント

- **[Docker環境のセットアップ](src/docker.md)** - Jupyter Notebookの起動手順
- **[Notebookの説明](src/notebooks.md)** - ノートブックの命名規則と使い方
- **[機械学習Tips](src/ml_tips.md)** - よく使うコードパターンとトラブルシューティング

## 🚀 クイックスタート

1. **Docker環境をセットアップ**
   ```bash
   # 詳細は src/docker.md を参照
   docker pull gcr.io/kaggle-images/python
   ```

2. **データを取得**
   ```bash
   # Kaggle APIを使用
   kaggle competitions download -c titanic -p competitions/titanic/data/input
   cd competitions/titanic/data/input && unzip titanic.zip && rm titanic.zip
   ```

3. **Notebookを起動**
   ```bash
   # 詳細は src/docker.md を参照
   docker run -it --rm -p 8888:8888 \
     -v /Users/orimotoseiya/Desktop/Kaggle:/workspace \
     -v ~/.kaggle:/root/.kaggle \
     gcr.io/kaggle-images/python bash
   ```

## 📖 使用している主なライブラリ

- **データ処理**: `pandas`, `numpy`
- **機械学習**: `scikit-learn`, `lightgbm`, `xgboost`, `catboost`
- **可視化**: `matplotlib`, `seaborn`, `ydata-profiling`
- **統計**: `scipy`

## 📌 今後の予定

- [ ] より高度な特徴量エンジニアリング
- [ ] アンサンブル手法の実装
- [ ] ハイパーパラメータチューニングの自動化
- [ ] 新しいコンペへの挑戦

---

**最終更新**: 2024年12月
