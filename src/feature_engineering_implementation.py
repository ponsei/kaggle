"""
特徴量エンジニアリング実装例
Playground Series S5E12 向け

必須・高優先度の特徴量エンジニアリング:
1. Label Encoding（既に実装済み）
2. Count Encoding
3. 比率特徴量（ldl_to_total_cholesterol, hdl_to_total_cholesterol, pulse_pressure）
"""

# ============================================================================
# 前提条件
# ============================================================================
# 以下の変数が定義されていることを前提とします:
# - train: 訓練データ（DataFrame）
# - test: テストデータ（DataFrame）
# - CATS: カテゴリ変数のリスト
# - BASE: ベース特徴量のリスト
# - NUMS: 数値変数のリスト

# ============================================================================
# 1. カウントエンコーディング（必須）
# ============================================================================
print("=" * 60)
print("📊 カウントエンコーディング")
print("=" * 60)

# カテゴリ変数の出現頻度を特徴量として追加
for col in CATS:
    # trainデータのみでカウントを計算（データリーク防止）
    count_map = train[col].value_counts().to_dict()
    
    # trainとtestの両方に適用
    train[f'{col}_count'] = train[col].map(count_map).fillna(0)
    test[f'{col}_count'] = test[col].map(count_map).fillna(0)
    
    print(f"✅ {col}_count を追加 (train: {train[f'{col}_count'].min():.0f}~{train[f'{col}_count'].max():.0f})")

print(f"\n✅ {len(CATS)}個のカウント特徴量を追加しました")

# ============================================================================
# 2. 比率特徴量（高優先度）
# ============================================================================
print("\n" + "=" * 60)
print("📊 比率特徴量の作成")
print("=" * 60)

# 2.1 コレステロール関連の比率
# LDLコレステロール / 総コレステロール
train['ldl_to_total_cholesterol'] = train['ldl_cholesterol'] / (train['cholesterol_total'] + 1e-6)
test['ldl_to_total_cholesterol'] = test['ldl_cholesterol'] / (test['cholesterol_total'] + 1e-6)
print("✅ ldl_to_total_cholesterol を追加")

# HDLコレステロール / 総コレステロール
train['hdl_to_total_cholesterol'] = train['hdl_cholesterol'] / (train['cholesterol_total'] + 1e-6)
test['hdl_to_total_cholesterol'] = test['hdl_cholesterol'] / (test['cholesterol_total'] + 1e-6)
print("✅ hdl_to_total_cholesterol を追加")

# 2.2 血圧関連の比率
# 脈圧（収縮期血圧 - 拡張期血圧）
train['pulse_pressure'] = train['systolic_bp'] - train['diastolic_bp']
test['pulse_pressure'] = test['systolic_bp'] - test['diastolic_bp']
print("✅ pulse_pressure を追加")

# 平均動脈圧（オプション: 追加で試す価値あり）
train['mean_arterial_pressure'] = train['diastolic_bp'] + (train['systolic_bp'] - train['diastolic_bp']) / 3
test['mean_arterial_pressure'] = test['diastolic_bp'] + (test['systolic_bp'] - test['diastolic_bp']) / 3
print("✅ mean_arterial_pressure を追加（オプション）")

print(f"\n✅ 比率特徴量を追加しました")

# ============================================================================
# 3. BASE特徴量の更新（新しく追加した特徴量を含める）
# ============================================================================
print("\n" + "=" * 60)
print("📊 特徴量リストの更新")
print("=" * 60)

# 新しく追加した特徴量を取得
new_features = []
for col in train.columns:
    if col not in BASE and col not in ['id', TARGET]:
        new_features.append(col)

# BASEに追加
BASE = BASE + new_features

print(f"✅ 新規特徴量: {len(new_features)}個")
print(f"   {new_features}")
print(f"\n✅ 更新後のBASE特徴量数: {len(BASE)}個")

# ============================================================================
# 4. 特徴量の確認（オプション）
# ============================================================================
print("\n" + "=" * 60)
print("📊 追加した特徴量の統計情報")
print("=" * 60)

# カウント特徴量の確認
count_features = [f'{col}_count' for col in CATS]
if count_features:
    print("\n【カウント特徴量】")
    print(train[count_features].describe().T)

# 比率特徴量の確認
ratio_features = ['ldl_to_total_cholesterol', 'hdl_to_total_cholesterol', 'pulse_pressure']
if all(col in train.columns for col in ratio_features):
    print("\n【比率特徴量】")
    print(train[ratio_features].describe().T)

print("\n" + "=" * 60)
print("✅ 特徴量エンジニアリング完了")
print("=" * 60)

