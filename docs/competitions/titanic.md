# Titanic - Machine Learning from Disaster

## 📋 コンペ概要

- **タスク**: 二値分類（生存予測）
- **評価指標**: Accuracy（正解率）
- **Kaggle URL**: https://www.kaggle.com/competitions/titanic

## 📊 データ概要

- **訓練データ**: `train.csv` (891行)
- **テストデータ**: `test.csv` (418行)
- **提出形式**: `PassengerId`, `Survived` (0 or 1)

### 特徴量

- **数値特徴量**: `Age`, `Fare`, `SibSp`, `Parch`
- **カテゴリ特徴量**: `Sex`, `Embarked`, `Pclass`
- **その他**: `Name`, `Ticket`, `Cabin`

## 📚 ノートブック

### 1. Titanic Kaggle.ipynb
基本的な実装：
- ロジスティック回帰
- ランダムフォレスト
- 基本的な前処理

### 2. Titanic_LightGBM.ipynb
LightGBMを使用した実装：
- LightGBM分類器
- 交差検証（KFold）
- loss曲線の可視化
- 特徴量重要度の可視化

### 3. Titanic Top Solution (Clean Version).ipynb
上位解法の実装：
- 高度な特徴量エンジニアリング
- アンサンブル手法

## 🔑 重要なポイント

### 前処理

```python
# 欠損値の補完
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# カテゴリ変数のエンコーディング
train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)
```

### 特徴量エンジニアリング

```python
# 家族サイズ
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

# 一人旅かどうか
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
```

## 📈 ベストスコア

- 現在の最高スコア: [記録を更新]

## 🚀 提出方法

```bash
kaggle competitions submit -c titanic \
  -f submissions/submission_titanic_lgbm.csv \
  -m "LightGBM with cross-validation"
```

---

**関連ドキュメント**: [README.md](../../README.md) | [機械学習Tips](../ml_tips.md)

