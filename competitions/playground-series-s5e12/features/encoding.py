"""
エンコーディング関連の処理
ラベルエンコーディング、順序エンコーディング、Target Encodingなど
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List


def label_encode_categorical(train: pd.DataFrame, test: pd.DataFrame, 
                             categorical_cols: List[str]) -> tuple:
    """
    カテゴリ変数をラベルエンコーディング
    
    Parameters:
    -----------
    train : pd.DataFrame
        訓練データ
    test : pd.DataFrame
        テストデータ
    categorical_cols : List[str]
        エンコーディングするカテゴリ変数のリスト
    
    Returns:
    --------
    tuple : (train, test, label_encoders)
        label_encoders: 各カラムのLabelEncoderの辞書
    """
    train = train.copy()
    test = test.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        if col not in train.columns:
            continue
        
        le = LabelEncoder()
        # 訓練データとテストデータを結合してfit
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        
        # 変換
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        
        label_encoders[col] = le
    
    return train, test, label_encoders


def ordinal_encode(train: pd.DataFrame, test: pd.DataFrame,
                   ordinal_mappings: Dict[str, Dict[str, int]]) -> tuple:
    """
    順序エンコーディング（カテゴリに順序がある場合）
    
    Parameters:
    -----------
    train : pd.DataFrame
        訓練データ
    test : pd.DataFrame
        テストデータ
    ordinal_mappings : Dict[str, Dict[str, int]]
        各カラムのカテゴリ→数値のマッピング
        例: {'education_level': {'Highschool': 1, 'Bachelor': 2, 'Master': 3}}
    
    Returns:
    --------
    tuple : (train, test)
    """
    train = train.copy()
    test = test.copy()
    
    for col, mapping in ordinal_mappings.items():
        if col in train.columns:
            train[col] = train[col].map(mapping).fillna(0)
        if col in test.columns:
            test[col] = test[col].map(mapping).fillna(0)
    
    return train, test


def target_encode(train: pd.DataFrame, test: pd.DataFrame,
                  categorical_cols: List[str], target_col: str,
                  smoothing: float = 1.0) -> tuple:
    """
    Target Encoding（目的変数との関係を反映したエンコーディング）
    
    Parameters:
    -----------
    train : pd.DataFrame
        訓練データ
    test : pd.DataFrame
        テストデータ
    categorical_cols : List[str]
        エンコーディングするカテゴリ変数のリスト
    target_col : str
        目的変数のカラム名
    smoothing : float
        スムージングパラメータ（デフォルト: 1.0）
    
    Returns:
    --------
    tuple : (train, test, encoding_maps)
        encoding_maps: 各カラムのエンコーディングマップの辞書
    """
    train = train.copy()
    test = test.copy()
    encoding_maps = {}
    
    global_mean = train[target_col].mean()
    
    for col in categorical_cols:
        if col not in train.columns:
            continue
        
        # 訓練データでエンコーディング値を計算
        agg = train.groupby(col)[target_col].agg(['mean', 'count'])
        
        # スムージング適用
        encoding_map = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        encoding_maps[col] = encoding_map.to_dict()
        
        # 訓練データに適用
        train[f'{col}_target_encoded'] = train[col].map(encoding_map).fillna(global_mean)
        
        # テストデータに適用（訓練データにないカテゴリは全体平均を使用）
        test[f'{col}_target_encoded'] = test[col].map(encoding_map).fillna(global_mean)
    
    return train, test, encoding_maps


def frequency_encode(train: pd.DataFrame, test: pd.DataFrame,
                    categorical_cols: List[str]) -> tuple:
    """
    頻度エンコーディング（カテゴリの出現頻度）
    
    Parameters:
    -----------
    train : pd.DataFrame
        訓練データ
    test : pd.DataFrame
        テストデータ
    categorical_cols : List[str]
        エンコーディングするカテゴリ変数のリスト
    
    Returns:
    --------
    tuple : (train, test, frequency_maps)
        frequency_maps: 各カラムの頻度マップの辞書
    """
    train = train.copy()
    test = test.copy()
    frequency_maps = {}
    
    for col in categorical_cols:
        if col not in train.columns:
            continue
        
        # 訓練データとテストデータを結合して頻度を計算
        combined = pd.concat([train[col], test[col]], axis=0)
        frequency_map = combined.value_counts().to_dict()
        frequency_maps[col] = frequency_map
        
        # 適用
        train[f'{col}_frequency'] = train[col].map(frequency_map)
        test[f'{col}_frequency'] = test[col].map(frequency_map)
    
    return train, test, frequency_maps
