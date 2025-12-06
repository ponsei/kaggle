# カテゴリエンコーディングと特徴量エンジニアリングのガイド
## Playground Series S5E12 (糖尿病診断予測) 向け

---

## 1. ラベルエンコーディング vs オーディナルエンコーディング

### 現在のデータ構造
- **カテゴリ変数**: `gender`, `ethnicity`, `education_level`, `income_level`, `smoking_status`, `employment_status`
- **カテゴリ数**: 3-5種類（比較的少ない）

### 推奨: **Label Encoding（現状維持）**

**理由:**
1. **LightGBMはカテゴリ変数を直接扱える**: `categorical_feature`パラメータで指定すれば、内部で最適な分割を学習します
2. **順序が不明確**: `education_level`や`income_level`に順序がある可能性はありますが、その順序が予測に有効かは不明
3. **実装が簡単**: 現在のコードで問題なく動作
4. **オーバーフィッティングのリスクが低い**: カテゴリ数が少ないため

### Ordinal Encodingを使う場合
- `education_level`や`income_level`に**明確な順序**があり、それが医学的に意味がある場合
- 例: `Low < Lower-Middle < Middle < Upper-Middle < High` のような順序

**実装例:**
```python
from sklearn.preprocessing import OrdinalEncoder

# 順序を明示的に定義
education_order = [['Highschool', 'Some College', 'Graduate', 'Postgraduate']]
income_order = [['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']]

ordinal_encoder = OrdinalEncoder(categories=education_order)
train['education_level_ordinal'] = ordinal_encoder.fit_transform(train[['education_level']])
```

**結論**: ベースラインでは**Label Encodingのまま**で問題ありません。後でOrdinal Encodingを試すのは良いアプローチです。

---

## 2. カウントエンコーディング

### 推奨: **実施する（追加特徴量として）**

**理由:**
1. **カテゴリの希少性が予測に有用**: 特定のカテゴリの出現頻度が糖尿病リスクと関連する可能性
2. **実装が簡単**: 計算コストが低い
3. **LightGBMと相性が良い**: 数値特徴量として扱える

**実装例:**
```python
# カウントエンコーディング
for col in CATS:
    # trainデータでカウントを計算
    count_map = train[col].value_counts().to_dict()
    # trainとtestの両方に適用
    train[f'{col}_count'] = train[col].map(count_map)
    test[f'{col}_count'] = test[col].map(count_map)
    # 新しいカテゴリ（testにのみ存在）は0にする
    train[f'{col}_count'] = train[f'{col}_count'].fillna(0)
    test[f'{col}_count'] = test[f'{col}_count'].fillna(0)
```

**注意点:**
- データリークを防ぐため、**trainデータのみでカウントを計算**し、それをtestに適用
- 新しいカテゴリ（testにのみ存在）は0または平均値で埋める

---

## 3. 有効な特徴量エンジニアリング（ベースライン向け）

### 3.1 数値変数間の比率特徴量（高優先度）

**なぜ比率特徴量を使うのか？**

#### 1. **医学的・生物学的な意味がある**
- **コレステロール比率**: `LDL/総コレステロール` や `HDL/総コレステロール` は、医学的に重要な指標です
  - 例: `LDL/総コレステロール` が高い = 悪玉コレステロールの割合が高い = 糖尿病リスクが高い可能性
- **血圧の比率**: 脈圧（収縮期 - 拡張期）は心血管疾患のリスク指標として使われます

#### 2. **相関の高い変数間の関係を明示化**
- `ldl_cholesterol` と `cholesterol_total` の相関は **0.81**（強い相関）
- モデルは2つの変数から「比率」を学習する必要があるが、**比率を直接与える**ことで：
  - 学習が効率化される
  - より解釈しやすい特徴量になる

#### 3. **スケール不変性（ロバスト性）**
- 比率は**スケールに依存しない**ため、外れ値の影響を受けにくい
- 例: `LDL=100, 総コレステロール=200` → 比率=0.5
- 例: `LDL=150, 総コレステロール=300` → 比率=0.5（同じ比率）
- モデルは「絶対値」ではなく「相対的な関係」を学習できる

#### 4. **多重共線性の回避と情報抽出**
- 相関の高い変数（`ldl_cholesterol` と `cholesterol_total`）をそのまま使うと：
  - 情報が重複する
  - 特徴量重要度の解釈が難しくなる
- 比率を作ることで：
  - **新しい情報**を抽出できる（例: 「LDLが総コレステロールに占める割合」）
  - 元の変数と比率の両方を使うことで、より豊富な情報を提供できる

#### 5. **モデルの学習効率**
- LightGBMのようなツリー系モデルは、比率を**1回の分割**で学習できる
- 元の変数だけだと、複数の分割が必要になる可能性がある
- 例: `if ldl_cholesterol / cholesterol_total > 0.6` という1つの条件で、複雑な関係を表現できる

#### 6. **実例: このデータセットでの効果**
```
元の特徴量:
- ldl_cholesterol = 120
- cholesterol_total = 200
- hdl_cholesterol = 50

比率特徴量を追加:
- ldl_to_total = 120/200 = 0.6 (60%がLDL)
- hdl_to_total = 50/200 = 0.25 (25%がHDL)

→ モデルは「LDLの割合が高い人」を直接識別できる
```

```python
# コレステロール関連の比率
train['ldl_to_total_cholesterol'] = train['ldl_cholesterol'] / (train['cholesterol_total'] + 1e-6)
test['ldl_to_total_cholesterol'] = test['ldl_cholesterol'] / (train['cholesterol_total'] + 1e-6)

train['hdl_to_total_cholesterol'] = train['hdl_cholesterol'] / (train['cholesterol_total'] + 1e-6)
test['hdl_to_total_cholesterol'] = test['hdl_cholesterol'] / (train['cholesterol_total'] + 1e-6)

# 血圧関連
train['pulse_pressure'] = train['systolic_bp'] - train['diastolic_bp']
test['pulse_pressure'] = test['systolic_bp'] - test['diastolic_bp']

train['mean_arterial_pressure'] = train['diastolic_bp'] + (train['systolic_bp'] - train['diastolic_bp']) / 3
test['mean_arterial_pressure'] = test['diastolic_bp'] + (test['systolic_bp'] - test['diastolic_bp']) / 3
```

### 3.2 年齢グループ化（中優先度）

**理由**: 年齢は糖尿病リスクと強い関連があるが、非線形関係の可能性

```python
# 年齢グループ（10歳刻み）
train['age_group'] = pd.cut(train['age'], bins=[0, 30, 40, 50, 60, 70, 100], labels=[1, 2, 3, 4, 5, 6])
test['age_group'] = pd.cut(test['age'], bins=[0, 30, 40, 50, 60, 70, 100], labels=[1, 2, 3, 4, 5, 6])
train['age_group'] = train['age_group'].astype(int)
test['age_group'] = test['age_group'].astype(int)
```

### 3.3 BMIカテゴリ化（中優先度）

**理由**: BMIは連続値だが、医学的なカテゴリ（正常、過体重、肥満）が定義されている

```python
# BMIカテゴリ（WHO基準）
def bmi_category(bmi):
    if bmi < 18.5:
        return 0  # Underweight
    elif bmi < 25:
        return 1  # Normal
    elif bmi < 30:
        return 2  # Overweight
    else:
        return 3  # Obese

train['bmi_category'] = train['bmi'].apply(bmi_category)
test['bmi_category'] = test['bmi'].apply(bmi_category)
```

### 3.4 相互作用特徴量（低優先度、試す価値あり）

**理由**: 複数のリスク要因の組み合わせが重要（例：高齢×高BMI）

```python
# 年齢 × BMI
train['age_bmi'] = train['age'] * train['bmi']
test['age_bmi'] = test['age'] * test['bmi']

# 運動 × スクリーンタイム（活動性の指標）
train['activity_balance'] = train['physical_activity_minutes_per_week'] / (train['screen_time_hours_per_day'] * 60 + 1)
test['activity_balance'] = test['physical_activity_minutes_per_week'] / (test['screen_time_hours_per_day'] * 60 + 1)
```

### 3.5 統計的特徴量（低優先度）

**理由**: 複数の関連変数の統計量が予測に有用な場合がある

```python
# コレステロール関連の統計量
cholesterol_cols = ['cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides']
train['cholesterol_mean'] = train[cholesterol_cols].mean(axis=1)
test['cholesterol_mean'] = test[cholesterol_cols].mean(axis=1)

# 血圧関連
bp_cols = ['systolic_bp', 'diastolic_bp']
train['bp_mean'] = train[bp_cols].mean(axis=1)
test['bp_mean'] = test[bp_cols].mean(axis=1)
```

---

## 4. 実装の優先順位（ベースライン）

### 必須（すぐに追加すべき）
1. ✅ **Label Encoding**（既に実装済み）
2. ✅ **Count Encoding**（追加推奨）

### 高優先度（効果が期待できる）
3. ✅ **比率特徴量**（`ldl_to_total_cholesterol`, `hdl_to_total_cholesterol`, `pulse_pressure`）

### 中優先度（試す価値あり）
4. ✅ **年齢グループ化**
5. ✅ **BMIカテゴリ化**

### 低優先度（時間があれば）
6. ⚠️ **相互作用特徴量**（過学習のリスクあり）
7. ⚠️ **統計的特徴量**（多重共線性のリスクあり）

---

## 5. 注意事項

### データリークの防止
- **Count Encoding**: trainデータのみでカウントを計算
- **統計的特徴量**: trainデータのみで平均・標準偏差を計算

### 多重共線性
- `ldl_cholesterol`と`cholesterol_total`は既に強い相関（0.81）がある
- 比率特徴量を追加することで、関係性を明示化できる
- LightGBMは多重共線性に強いが、解釈時は注意

### 過学習のリスク
- 相互作用特徴量は増やしすぎると過学習のリスクが高まる
- ベースラインでは最小限に留める

---

## 6. 実装コード例（統合版）

```python
# ============================================================================
# 1. カウントエンコーディング
# ============================================================================
for col in CATS:
    count_map = train[col].value_counts().to_dict()
    train[f'{col}_count'] = train[col].map(count_map).fillna(0)
    test[f'{col}_count'] = test[col].map(count_map).fillna(0)

# ============================================================================
# 2. 比率特徴量
# ============================================================================
# コレステロール関連
train['ldl_to_total'] = train['ldl_cholesterol'] / (train['cholesterol_total'] + 1e-6)
test['ldl_to_total'] = test['ldl_cholesterol'] / (test['cholesterol_total'] + 1e-6)

train['hdl_to_total'] = train['hdl_cholesterol'] / (train['cholesterol_total'] + 1e-6)
test['hdl_to_total'] = test['hdl_cholesterol'] / (test['cholesterol_total'] + 1e-6)

# 血圧関連
train['pulse_pressure'] = train['systolic_bp'] - train['diastolic_bp']
test['pulse_pressure'] = test['systolic_bp'] - test['diastolic_bp']

# ============================================================================
# 3. 年齢グループ化
# ============================================================================
train['age_group'] = pd.cut(train['age'], bins=[0, 30, 40, 50, 60, 70, 100], labels=[1, 2, 3, 4, 5, 6])
test['age_group'] = pd.cut(test['age'], bins=[0, 30, 40, 50, 60, 70, 100], labels=[1, 2, 3, 4, 5, 6])
train['age_group'] = train['age_group'].astype(int)
test['age_group'] = test['age_group'].astype(int)

# ============================================================================
# 4. BMIカテゴリ化
# ============================================================================
def bmi_category(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    else:
        return 3

train['bmi_category'] = train['bmi'].apply(bmi_category)
test['bmi_category'] = test['bmi'].apply(bmi_category)

# ============================================================================
# BASE特徴量の更新（新しく追加した特徴量を含める）
# ============================================================================
BASE = [col for col in train.columns if col not in EXCLUDE]
NUMS = [col for col in BASE if col not in CATS]
```

---

## まとめ

1. **エンコーディング**: Label Encodingで問題なし（現状維持）
2. **Count Encoding**: 追加推奨（簡単で効果的）
3. **特徴量エンジニアリング**: 比率特徴量（コレステロール、血圧）を最優先で追加
4. **段階的アプローチ**: まずはCount Encodingと比率特徴量を追加してベースラインを改善

