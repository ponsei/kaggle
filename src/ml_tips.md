# 機械学習Tips

よく使うコードパターンとトラブルシューティング集です。

## 📚 基本パターン

### 1. pandas で前処理

```python
import pandas as pd
import numpy as np

# データ読み込み
df = pd.read_csv('../input/titanic/train.csv')

# 欠損値の確認
print(df.isnull().sum())

# 欠損値の補完（中央値）
df['Age'] = df['Age'].fillna(df['Age'].median())

# 欠損値の補完（最頻値）
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# カテゴリ変数のエンコーディング（One-Hot）
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 数値特徴量の選択
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
```

### 2. scikit-learn で学習

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# データ分割
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# モデル学習
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# 予測と評価
y_pred = model.predict(X_valid)
print('Accuracy:', accuracy_score(y_valid, y_pred))
print(classification_report(y_valid, y_pred))
```

### 3. matplotlib/seaborn で可視化

```python
import matplotlib.pyplot as plt
import seaborn as sns

# スタイル設定
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 目的変数の分布
sns.countplot(x=y)
plt.title('Target Distribution')
plt.show()

# 特徴量重要度
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.show()

# 相関行列
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
```

## 🚀 高度なパターン

### LightGBM（分類）

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# データ分割
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 学習経過を記録
evals_result = {}

# モデル定義
gbm = lgb.LGBMClassifier(
    objective='binary',
    importance_type='gain',
    n_estimators=1000
)

# 学習
gbm.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_names=['train', 'valid'],
    eval_metric='binary_logloss',
    callbacks=[
        lgb.early_stopping(20),      # 20回改善しなければ打ち切り
        lgb.log_evaluation(0),       # ログ不要なら0
        lgb.record_evaluation(evals_result),  # 結果を記録
    ],
)

# loss曲線のプロット
plt.figure(figsize=(8, 4))
plt.plot(evals_result['train']['binary_logloss'], label='train_loss')
plt.plot(evals_result['valid']['binary_logloss'], label='valid_loss')
plt.xlabel('Iteration')
plt.ylabel('binary_logloss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 予測
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
```

### 交差検証（KFold）

```python
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
score_list = []
models = []

for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
    X_train_fold = X.iloc[train_index]
    X_valid_fold = X.iloc[valid_index]
    y_train_fold = y.iloc[train_index]
    y_valid_fold = y.iloc[valid_index]
    
    print(f'Fold {fold + 1} start')
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_fold, y_train_fold)
    
    y_pred = model.predict(X_valid_fold)
    score = accuracy_score(y_valid_fold, y_pred)
    score_list.append(score)
    models.append(model)
    
    print(f'Fold {fold + 1} score: {score:.4f}')

print(f'Average score: {np.mean(score_list):.4f}')
```

### 提出ファイルの作成

```python
# テストデータで予測
test_pred = model.predict(X_test)

# 提出用DataFrame作成
# 注意: testデータからPassengerIdを取得（前処理で削除されていない場合）
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],  # 元のtestデータから取得
    'Survived': test_pred
})

# CSVとして保存
submission.to_csv('../submissions/submission.csv', index=False)
print(submission.head())
```

## 🔧 トラブルシューティング

### CSV読み込みエラー（`OSError: [Errno 35] Resource deadlock avoided`）

**原因**: pandasのCエンジンとファイルシステムの相性問題

**解決策1**: Pythonエンジンを使用
```python
pd.read_csv('path/to/file.csv', engine='python')
```

**解決策2**: csvモジュールを使用
```python
import csv
import pandas as pd

rows = []
with open('path/to/file.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)
df = pd.DataFrame(rows)
```

### LightGBMの早期停止エラー

**エラー**: `TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'`

**原因**: 新しいバージョンでは `early_stopping_rounds` が非推奨

**解決策**: `callbacks` を使用
```python
# 古い書き方（非推奨）
# gbm.fit(..., early_stopping_rounds=20)

# 新しい書き方
gbm.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[
        lgb.early_stopping(20),
        lgb.log_evaluation(0)
    ]
)
```

### 提出ファイルで `KeyError: 'PassengerId'`

**原因**: 前処理で `PassengerId` が削除された

**解決策**: 元のtestデータを再読み込み
```python
# 元のtestデータを再読み込み
test_org = pd.read_csv('../input/titanic/test.csv')
submission = pd.DataFrame({
    'PassengerId': test_org['PassengerId'],
    'Survived': test_pred
})
```

### `NameError: name 'X_test' is not defined`

**原因**: `X_test` が定義されていない

**解決策**: 特徴量を明示的に定義
```python
# 特徴量の定義
feature_cols = [c for c in train.columns if c not in ['PassengerId', 'Survived']]
X_train = train[feature_cols]
X_test = test[feature_cols]  # これが必要
```

### `DeprecationWarning: import pandas_profiling`

**原因**: `pandas_profiling` が非推奨

**解決策**: `ydata_profiling` を使用
```python
# 古い書き方
# from pandas_profiling import ProfileReport

# 新しい書き方
from ydata_profiling import ProfileReport
```

## 💡 よく使う便利コード

### データの基本情報を一括表示

```python
# pandasの表示オプション設定
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 基本情報
train_df.info()

# 統計サマリー（左揃え）
stats_df = pd.DataFrame(train_df[target_col].describe(), columns=[target_col])
display(stats_df.style.set_properties(**{'text-align': 'left'}))
```

### 欠損値の可視化

```python
# 欠損値の多い特徴量（上位10個）
missing = train_df.isnull().sum().sort_values(ascending=False).head(10)
if missing.sum() > 0:
    missing[missing > 0].plot(kind='barh', figsize=(8, 6))
    plt.title('欠損値の多い特徴量（上位10個）')
    plt.show()
```

### 特徴量重要度の可視化

```python
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(20).plot(kind='barh', figsize=(10, 8))
plt.title('Feature Importances (Top 20)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
```

---

**関連ドキュメント**: [README.md](../README.md) | [Notebookの説明](notebooks.md)

