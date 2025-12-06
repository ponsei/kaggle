# House Prices: Advanced Regression Techniques

## 📋 コンペ概要

- **タスク**: 回帰（住宅価格予測）
- **評価指標**: RMSE（Root Mean Squared Error）
- **Kaggle URL**: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## 📊 データ概要

- **訓練データ**: `train.csv` (1460行)
- **テストデータ**: `test.csv` (1459行)
- **目的変数**: `SalePrice` (住宅価格)
- **特徴量**: 80以上（数値・カテゴリ混在）

### 主な特徴量

- **数値特徴量**: `LotArea`, `GrLivArea`, `TotalBsmtSF`, `YearBuilt` など
- **カテゴリ特徴量**: `MSZoning`, `Neighborhood`, `HouseStyle` など

## 📚 ノートブック

### 1. House_Prices_Starter.ipynb
初心者向け解説付き：
- 回帰問題の説明
- RMSEの説明
- 基本的な前処理
- 複数モデルの比較（Linear Regression, RandomForest, GradientBoosting）

### 2. House_Prices_Comprehensive_EDA.ipynb
包括的なEDA：
- 目的変数の分布（正規分布、対数正規分布との比較）
- 数値特徴量の分析
- カテゴリ特徴量の分析
- 相関行列
- 外れ値の検出
- train/test分布の比較

## 🔑 重要なポイント

### 目的変数の分布

`SalePrice`は右に歪んだ分布（対数正規分布に近い）：
```python
# 対数変換が有効
y_log = np.log1p(train_df['SalePrice'])
```

### 前処理

```python
# 欠損値の補完
# 数値特徴量: 中央値
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    train_df[col] = train_df[col].fillna(train_df[col].median())

# カテゴリ特徴量: 最頻値
categorical_cols = train_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
```

### 特徴量エンジニアリング

```python
# 総面積
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']

# 部屋数/面積
train_df['RoomsPerArea'] = train_df['TotRmsAbvGrd'] / train_df['GrLivArea']

# 築年数
train_df['Age'] = train_df['YrSold'] - train_df['YearBuilt']

# リノベーション年数
train_df['RemodAge'] = train_df['YrSold'] - train_df['YearRemodAdd']
```

### 外れ値の処理

```python
# GrLivAreaとSalePriceの散布図で外れ値を確認
# 通常、GrLivArea > 4000 のデータは外れ値として扱う
train_df = train_df[train_df['GrLivArea'] < 4000]
```

## 📈 ベストスコア

- 現在の最高スコア: [記録を更新]

## 🚀 提出方法

```bash
kaggle competitions submit -c house-prices-advanced-regression-techniques \
  -f submissions/submission_house_prices.csv \
  -m "First submission"
```

---

**関連ドキュメント**: [README.md](../../README.md) | [機械学習Tips](../ml_tips.md)

