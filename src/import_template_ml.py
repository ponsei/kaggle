"""
機械学習用ライブラリ読み込みテンプレート
モデル学習に特化したバージョン
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 機械学習
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

# 高度なモデル
import lightgbm as lgb
import xgboost as xgb
# import catboost as cb  # 必要に応じて

# 可視化（必要に応じて）
import matplotlib.pyplot as plt
import seaborn as sns

# 乱数シード設定
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

