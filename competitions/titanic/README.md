# Titanic - Machine Learning from Disaster

## 📋 コンペ概要

- **タスク**: 二値分類（生存予測）
- **評価指標**: Accuracy（正解率）
- **Kaggle URL**: https://www.kaggle.com/competitions/titanic

## 📁 ディレクトリ構成

```
titanic/
├── configs/           # 設定ファイル
│   └── default.json
├── data/
│   ├── input/         # オリジナルの生データ（train.csv, test.csv）
│   └── output/        # 処理されたデータや予測結果
├── features/          # 特徴量エンジニアリング関連ファイル
├── logs/              # 実行ログやモデルの学習ログ
├── notebooks/         # Jupyter Notebook
└── submissions/       # 提出ファイル（CSV）
```

## 📚 ノートブック

- `Titanic Kaggle.ipynb` - 基本実装（ロジスティック回帰、ランダムフォレスト）
- `Titanic_LightGBM.ipynb` - LightGBMを使用した実装（交差検証、loss曲線の可視化）
- `Titanic Top Solution (Clean Version).ipynb` - 上位解法の実装
- `Titanic_TFDF_Step_by_Step.ipynb` - TensorFlow Decision Forestsの実装

## 🚀 提出方法

```bash
kaggle competitions submit -c titanic \
  -f competitions/titanic/submissions/submission.csv \
  -m "Submission message"
```
