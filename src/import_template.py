"""
ライブラリ読み込みテンプレート
Kaggleコンペティションでよく使うライブラリをまとめています
"""

# ============================================================================
# 基本ライブラリ
# ============================================================================
import os
import sys
import warnings
from pathlib import Path

# 警告を非表示にする
warnings.filterwarnings('ignore')

# ============================================================================
# データ処理
# ============================================================================
import pandas as pd
import numpy as np

# pandasの表示オプション設定
pd.set_option('display.max_columns', None)  # すべての列を表示
pd.set_option('display.max_rows', 100)      # 最大100行まで表示
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')  # 数値を2桁で表示

# ============================================================================
# 可視化
# ============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# スタイル設定
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# 日本語フォントの設定（必要に応じて）
# plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================================
# 統計・数学
# ============================================================================
import scipy.stats as stats
from scipy import stats as scipy_stats

# ============================================================================
# 機械学習（scikit-learn）
# ============================================================================
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# 分類モデル
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# 回帰モデル
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

# ============================================================================
# 高度な機械学習モデル
# ============================================================================
try:
    import lightgbm as lgb
except ImportError:
    print("⚠️ LightGBMがインストールされていません: pip install lightgbm")

try:
    import xgboost as xgb
except ImportError:
    print("⚠️ XGBoostがインストールされていません: pip install xgboost")

try:
    import catboost as cb
except ImportError:
    print("⚠️ CatBoostがインストールされていません: pip install catboost")

# ============================================================================
# その他の便利なライブラリ
# ============================================================================
try:
    from ydata_profiling import ProfileReport
except ImportError:
    print("⚠️ ydata-profilingがインストールされていません: pip install ydata-profiling")

try:
    from tqdm import tqdm
    tqdm.pandas()  # pandasにプログレスバーを追加
except ImportError:
    print("⚠️ tqdmがインストールされていません: pip install tqdm")

# ============================================================================
# 便利関数
# ============================================================================
def set_random_seed(seed=42):
    """乱数シードを設定"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# デフォルトでシードを設定
set_random_seed(42)

# ============================================================================
# パス設定（必要に応じて）
# ============================================================================
# INPUT_DIR = '../input/playground-series-s5e12/'
# OUTPUT_DIR = '../output/'
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 確認用
# ============================================================================
if __name__ == "__main__":
    print("✅ ライブラリの読み込みが完了しました")
    print(f"pandas version: {pd.__version__}")
    print(f"numpy version: {np.__version__}")
    print(f"matplotlib version: {plt.matplotlib.__version__}")
    print(f"seaborn version: {sns.__version__}")

