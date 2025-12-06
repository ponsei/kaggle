# BNP Paribas Cardif Claims Management

## 📋 コンペ概要

- **タスク**: 二値分類（保険請求の管理）
- **評価指標**: Log Loss（対数損失）
- **Kaggle URL**: https://www.kaggle.com/competitions/bnp-paribas-cardif-claims-management

## 📊 データ概要

- **訓練データ**: `train.csv`
- **テストデータ**: `test.csv`
- **特徴量**: 多数の数値特徴量とカテゴリ特徴量

### 特徴

- 多くの特徴量（100以上）
- 欠損値が多い
- 不均衡データの可能性

## 📚 ノートブック

### BNP_Paribas_Cardif_Starter.ipynb
初心者向け解説付き：
- コンペの要点説明
- 機械学習用語の解説
- 基本的な前処理
- 複数モデルの比較（Logistic Regression, RandomForest, GradientBoosting）

## 🔑 重要なポイント

### Log Lossについて

Log Lossは確率予測の精度を評価する指標です：
- 0に近いほど良い（完全予測で0）
- 予測確率が重要（0/1だけでなく、確率も評価される）

### 前処理

```python
# 欠損値の補完（中央値）
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    train_df[col] = train_df[col].fillna(train_df[col].median())

# カテゴリ変数のエンコーディング
categorical_cols = train_df.select_dtypes(include=['object']).columns
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
```

## 📈 ベストスコア

- 現在の最高スコア: [記録を更新]

## 🚀 提出方法

```bash
kaggle competitions submit -c bnp-paribas-cardif-claims-management \
  -f submissions/submission_bnp.csv \
  -m "First submission"
```

---

**関連ドキュメント**: [README.md](../../README.md) | [機械学習Tips](../ml_tips.md)

