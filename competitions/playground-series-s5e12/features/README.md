# 特徴量エンジニアリングモジュール

このディレクトリには、再利用可能な特徴量エンジニアリング関数が含まれています。

## 📁 ファイル構成

- `base.py` - ベース特徴量の処理と分類
- `statistical.py` - 統計的特徴量（コレステロール比率、血圧関連、生活習慣スコアなど）
- `interaction.py` - 相互作用特徴量（高重要度特徴量同士の組み合わせなど）
- `encoding.py` - エンコーディング処理（ラベル、順序、Target、頻度）

## 🚀 使用方法

### ノートブックでの使用例

```python
import sys
sys.path.append('..')  # featuresディレクトリにアクセスするため

from features import (
    get_base_features,
    create_all_statistical_features,
    create_all_interaction_features,
    label_encode_categorical,
    print_feature_summary
)

# 1. ベース特徴量の取得
feature_dict = get_base_features(train, test, exclude_cols=['id', 'diagnosed_diabetes'])
print_feature_summary(feature_dict)

BASE = feature_dict['BASE']
CATS = feature_dict['CATS']
NUMS = feature_dict['NUMS']

# 2. 統計的特徴量の作成
train, test = create_all_statistical_features(train, test)

# 3. 相互作用特徴量の作成
train, test = create_all_interaction_features(train, test)

# 4. エンコーディング
train, test, label_encoders = label_encode_categorical(
    train, test, 
    categorical_cols=['gender', 'ethnicity', 'employment_status']
)

# 5. 更新された特徴量リストを取得
updated_BASE = [col for col in train.columns if col not in ['id', 'diagnosed_diabetes']]
```

## 📝 各モジュールの詳細

### base.py
- `get_base_features()`: ベース特徴量を分類（カテゴリ変数、数値変数）
- `print_feature_summary()`: 特徴量のサマリーを表示

### statistical.py
- `create_cholesterol_features()`: コレステロール関連の統計的特徴量
- `create_blood_pressure_features()`: 血圧関連の統計的特徴量
- `create_lifestyle_features()`: 生活習慣スコア
- `create_age_features()`: 年齢関連の特徴量
- `create_bmi_features()`: BMI関連の特徴量
- `create_all_statistical_features()`: すべての統計的特徴量を一度に作成

### interaction.py
- `create_high_importance_interactions()`: 高重要度特徴量の相互作用
- `create_cholesterol_interactions()`: コレステロール値同士の相互作用
- `create_lifestyle_interactions()`: 生活習慣の相互作用
- `create_demographic_interactions()`: 人口統計学的特徴量の相互作用
- `create_all_interaction_features()`: すべての相互作用特徴量を一度に作成

### encoding.py
- `label_encode_categorical()`: ラベルエンコーディング
- `ordinal_encode()`: 順序エンコーディング
- `target_encode()`: Target Encoding（目的変数との関係を反映）
- `frequency_encode()`: 頻度エンコーディング

## 💡 カスタマイズ

各関数は独立しているため、必要な特徴量のみを選択的に使用できます。

例：
```python
# 統計的特徴量の一部のみ使用
train = create_cholesterol_features(train)
test = create_cholesterol_features(test)

train = create_blood_pressure_features(train)
test = create_blood_pressure_features(test)
```

## 🔄 更新履歴

- 2024-12-08: 初版作成
