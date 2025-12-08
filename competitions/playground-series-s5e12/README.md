# Playground Series S5E12 - Diabetes Prediction Challenge

## 📋 コンペ概要

- **タスク**: 二値分類（糖尿病診断予測）
- **評価指標**: Log Loss（対数損失）
- **Kaggle URL**: https://www.kaggle.com/competitions/playground-series-s5e12

## 📁 ディレクトリ構成

```
playground-series-s5e12/
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

- `Playground Series S5E12.ipynb` - メインノートブック

## 🚀 提出方法

```bash
kaggle competitions submit -c playground-series-s5e12 \
  -f competitions/playground-series-s5e12/submissions/submission.csv \
  -m "Submission message"
```
