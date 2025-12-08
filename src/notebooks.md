# Notebookの説明

このプロジェクトで使用しているJupyter Notebookの命名規則と使い方について説明します。

## 📝 命名規則

### 基本パターン

- `[コンペ名]_Starter.ipynb` - 初心者向け解説付きの入門ノートブック
- `[コンペ名]_[手法名].ipynb` - 特定の手法やモデルを使用したノートブック
- `[コンペ名]_Comprehensive_EDA.ipynb` - 包括的な探索的データ分析（EDA）

### 具体例

- `Titanic Kaggle.ipynb` - Titanicコンペの基本実装
- `Titanic_LightGBM.ipynb` - LightGBMを使用した実装
- `House_Prices_Starter.ipynb` - House Pricesコンペの初心者向け解説付き
- `House_Prices_Comprehensive_EDA.ipynb` - House Pricesの包括的なEDA

## 📚 ノートブックの構成

### 1. ライブラリのインポート

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

### 2. データの読み込み

```python
# Docker環境用のパス
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
```

### 3. EDA（探索的データ分析）

- 基本情報の確認（`info()`, `describe()`）
- 欠損値の確認
- 目的変数の分布
- 特徴量間の相関
- 可視化

### 4. 前処理

- 欠損値の補完
- カテゴリ変数のエンコーディング
- 特徴量エンジニアリング
- スケーリング

### 5. モデル学習

- データの分割（train/valid）
- モデルの定義と学習
- 予測と評価

### 6. 提出ファイルの作成

```python
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
submission.to_csv('../submissions/submission.csv', index=False)
```

## 🎯 ノートブック一覧

### Titanic

- **Titanic Kaggle.ipynb**: 基本的な実装（ロジスティック回帰、ランダムフォレスト）
- **Titanic_LightGBM.ipynb**: LightGBMを使用した実装（交差検証、loss曲線の可視化）
- **Titanic Top Solution (Clean Version).ipynb**: 上位解法の実装

### BNP Paribas Cardif

- **BNP_Paribas_Cardif_Starter.ipynb**: 初心者向け解説付き（二値分類、Log Loss）

### House Prices

- **House_Prices_Starter.ipynb**: 初心者向け解説付き（回帰、RMSE）
- **House_Prices_Comprehensive_EDA.ipynb**: 包括的なEDA（分布、相関、外れ値）

### atmaCup#8

- **atmaCup#8.ipynb**: atmaCup#8の実装

## 💡 ベストプラクティス

### 1. セルの実行順序

- 上から順に実行する
- エラーが出た場合は、該当セル以前を再実行

### 2. 変数名の一貫性

```python
# 推奨
train_df, test_df  # DataFrame
X_train, y_train   # 特徴量と目的変数
model              # モデル
```

### 3. コメントとMarkdownセル

- 重要な処理にはコメントを追加
- セクションごとにMarkdownセルで説明を追加

### 4. データの保存

- 中間結果は必要に応じて保存
- 提出ファイルは `submissions/` ディレクトリに保存

## 🔄 ノートブックの更新

新しいノートブックを作成する際は：

1. 命名規則に従ってファイル名を決定
2. 既存のノートブックを参考にする
3. 必要に応じて `_Starter` 版を作成（初心者向け解説付き）

---

**関連ドキュメント**: [README.md](../README.md) | [機械学習Tips](ml_tips.md)

