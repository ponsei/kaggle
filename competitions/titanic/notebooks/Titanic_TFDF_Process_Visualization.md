# Titanic TensorFlow Decision Forests ノートブック - 手順可視化

## 📊 全体フローチャート

```mermaid
graph TD
    A[開始] --> B[1. ライブラリのインポート]
    B --> C[2. データの読み込み]
    C --> D[3. データ探索 EDA]
    D --> E[4. データ前処理]
    E --> F[5. 特徴量エンジニアリング]
    F --> G[6. データセット変換]
    G --> H[7. モデル構築]
    H --> I[8. モデル訓練]
    I --> J[9. モデル評価]
    J --> K[10. 予測生成]
    K --> L[11. モデル可視化]
    L --> M[終了]
    
    style A fill:#e1f5ff
    style M fill:#e1f5ff
    style H fill:#fff4e1
    style I fill:#fff4e1
    style J fill:#fff4e1
```

## 🔍 詳細手順

### ステップ1: ライブラリのインポート

```python
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

**目的**: 必要なライブラリを準備

---

### ステップ2: データの読み込み

```python
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
```

**データ構造**:
- `train.csv`: 891行 × 12列（Survived含む）
- `test.csv`: 418行 × 11列（Survivedなし）

---

### ステップ3: データ探索（EDA）

```mermaid
graph LR
    A[データ読み込み] --> B[基本情報確認]
    B --> C[欠損値確認]
    C --> D[統計情報確認]
    D --> E[可視化]
    
    B --> B1[.info]
    B --> B2[.head]
    B --> B3[.shape]
    
    C --> C1[.isnull.sum]
    C --> C2[欠損率計算]
    
    D --> D1[.describe]
    D --> D2[.value_counts]
    
    E --> E1[ヒストグラム]
    E --> E2[相関行列]
    E --> E3[箱ひげ図]
```

**実行コード例**:
```python
# 基本情報
train_df.info()
train_df.describe()

# 欠損値確認
train_df.isnull().sum()

# 可視化
sns.histplot(train_df['Age'].dropna(), bins=30)
plt.title('Age Distribution')
plt.show()
```

---

### ステップ4: データ前処理

```mermaid
graph TD
    A[生データ] --> B[欠損値処理]
    B --> C[カテゴリカル変数処理]
    C --> D[数値変数処理]
    D --> E[前処理済みデータ]
    
    B --> B1[Age: 中央値で補完]
    B --> B2[Embarked: 最頻値で補完]
    B --> B3[Cabin: 削除 or 新特徴量]
    
    C --> C1[Sex: 0/1変換]
    C --> C2[Embarked: ダミー変数]
    
    D --> D1[Fare: 正規化 or そのまま]
    D --> D2[Pclass: そのまま or エンコーディング]
```

**実行コード例**:
```python
# 欠損値補完
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# カテゴリカル変数エンコーディング
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df = pd.get_dummies(train_df, columns=['Embarked'], prefix='Emb')
```

---

### ステップ5: 特徴量エンジニアリング

```mermaid
graph TD
    A[元の特徴量] --> B[新特徴量作成]
    B --> C[特徴量選択]
    C --> D[最終特徴量セット]
    
    B --> B1[Title: Nameから抽出]
    B --> B2[FamilySize: SibSp + Parch + 1]
    B --> B3[IsAlone: FamilySize == 1]
    B --> B4[AgeGroup: Ageをグループ化]
    B --> B5[FarePerPerson: Fare / FamilySize]
    
    C --> C1[不要な列削除]
    C --> C2[相関の高い特徴量削除]
```

**実行コード例**:
```python
# Title抽出
train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
train_df['Title'] = train_df['Title'].map(title_mapping)

# FamilySize
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
```

---

### ステップ6: データセット変換

```mermaid
graph LR
    A[Pandas DataFrame] --> B[pd_dataframe_to_tf_dataset]
    B --> C[TensorFlow Dataset]
    
    A --> A1[train_df]
    A --> A2[test_df]
    
    C --> C1[train_ds]
    C --> C2[test_ds]
```

**実行コード**:
```python
# ラベル列を指定して変換
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    train_df.drop('Survived', axis=1), 
    label='Survived'
)

# テストデータ（ラベルなし）
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    test_df
)
```

---

### ステップ7: モデル構築

```mermaid
graph TD
    A[モデル選択] --> B[ハイパーパラメータ設定]
    B --> C[モデルインスタンス作成]
    C --> D[コンパイル]
    
    A --> A1[RandomForestModel]
    A --> A2[GradientBoostedTreesModel]
    A --> A3[CARTModel]
    
    B --> B1[num_trees: 100]
    B --> B2[max_depth: 16]
    B --> B3[min_examples: 5]
```

**実行コード**:
```python
# ランダムフォレストモデル
model = tfdf.keras.RandomForestModel(
    num_trees=100,
    max_depth=16,
    min_examples=5,
    task=tfdf.keras.Task.CLASSIFICATION
)

# コンパイル（メトリクス指定）
model.compile(metrics=['accuracy'])
```

---

### ステップ8: モデル訓練

```mermaid
graph LR
    A[訓練データ] --> B[model.fit]
    B --> C[訓練済みモデル]
    
    B --> B1[エポック数]
    B --> B2[バリデーション分割]
    B --> B3[コールバック]
```

**実行コード**:
```python
# モデル訓練
model.fit(train_ds)

# または、バリデーション分割あり
model.fit(
    train_ds,
    validation_split=0.2,
    verbose=1
)
```

---

### ステップ9: モデル評価

```mermaid
graph TD
    A[訓練済みモデル] --> B[テストデータ評価]
    B --> C[メトリクス計算]
    C --> D[結果表示]
    
    C --> C1[Accuracy]
    C --> C2[Precision]
    C --> C3[Recall]
    C --> C4[F1-Score]
```

**実行コード**:
```python
# 評価
evaluation = model.evaluate(test_ds, return_dict=True)
print(f"Test Accuracy: {evaluation['accuracy']:.4f}")

# 予測
predictions = model.predict(test_ds)
predictions_binary = (predictions > 0.5).astype(int)
```

---

### ステップ10: 予測生成と提出

```mermaid
graph LR
    A[テストデータ] --> B[予測実行]
    B --> C[バイナリ変換]
    C --> D[提出ファイル作成]
    D --> E[CSV出力]
```

**実行コード**:
```python
# 予測
predictions = model.predict(test_ds)
predictions_binary = (predictions > 0.5).astype(int).flatten()

# 提出ファイル作成
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions_binary
})

submission.to_csv('submission.csv', index=False)
```

---

### ステップ11: モデル可視化

```mermaid
graph TD
    A[訓練済みモデル] --> B[決定木可視化]
    A --> C[特徴量重要度]
    A --> D[モデル統計]
    
    B --> B1[特定の木を表示]
    B --> B2[深さ制限]
    
    C --> C1[重要度ランキング]
    C --> C2[重要度プロット]
```

**実行コード**:
```python
# 決定木の可視化
tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0, max_depth=3)

# 特徴量重要度
importances = model.make_inspector().variable_importances()
print(importances)

# 統計情報
inspector = model.make_inspector()
print(inspector.num_trees())
print(inspector.evaluation())
```

---

## 📈 データフロー全体図

```mermaid
graph TB
    subgraph "データ準備フェーズ"
        A1[CSVファイル] --> A2[Pandas DataFrame]
        A2 --> A3[EDA・探索]
        A3 --> A4[前処理]
        A4 --> A5[特徴量エンジニアリング]
    end
    
    subgraph "モデル構築フェーズ"
        A5 --> B1[TF Dataset変換]
        B1 --> B2[モデル定義]
        B2 --> B3[モデル訓練]
    end
    
    subgraph "評価・予測フェーズ"
        B3 --> C1[モデル評価]
        B3 --> C2[予測生成]
        C2 --> C3[提出ファイル]
        B3 --> C4[モデル可視化]
    end
    
    style A1 fill:#e1f5ff
    style B2 fill:#fff4e1
    style B3 fill:#fff4e1
    style C3 fill:#e8f5e9
```

## 🎯 重要なポイント

1. **TF-DFの利点**: カテゴリカル変数を自動処理、前処理が簡単
2. **特徴量エンジニアリング**: Title、FamilySizeなどが重要
3. **モデル選択**: RandomForestModelが一般的に良い性能
4. **可視化**: 決定木の構造を確認して解釈性を確保

## 📝 典型的なコード構造

```python
# ============================================
# 1. インポート
# ============================================
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd

# ============================================
# 2. データ読み込み
# ============================================
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

# ============================================
# 3. 前処理
# ============================================
# 欠損値処理、特徴量エンジニアリングなど

# ============================================
# 4. データセット変換
# ============================================
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    train_df.drop('Survived', axis=1), 
    label='Survived'
)

# ============================================
# 5. モデル構築・訓練
# ============================================
model = tfdf.keras.RandomForestModel()
model.compile(metrics=['accuracy'])
model.fit(train_ds)

# ============================================
# 6. 予測・提出
# ============================================
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)
predictions = model.predict(test_ds)
# 提出ファイル作成...
```

