"""
ベース特徴量の処理
基本的な前処理と特徴量の分類
"""
import pandas as pd
import numpy as np


def get_base_features(train: pd.DataFrame, test: pd.DataFrame, exclude_cols: list = None) -> dict:
    """
    ベース特徴量を取得
    
    Parameters:
    -----------
    train : pd.DataFrame
        訓練データ
    test : pd.DataFrame
        テストデータ
    exclude_cols : list
        除外するカラム（デフォルト: ['id', 'diagnosed_diabetes']）
    
    Returns:
    --------
    dict : {
        'BASE': ベース特徴量のリスト,
        'CATS': カテゴリ変数のリスト,
        'NUMS': 数値変数のリスト,
        'train': 処理後の訓練データ,
        'test': 処理後のテストデータ
    }
    """
    if exclude_cols is None:
        exclude_cols = ['id', 'diagnosed_diabetes']
    
    # ベース特徴量（除外カラム以外）
    BASE = [col for col in train.columns if col not in exclude_cols]
    
    # カテゴリ変数（文字列型）
    CATS = train[BASE].select_dtypes(include=['object']).columns.tolist()
    
    # 数値変数（カテゴリ変数以外）
    NUMS = [col for col in BASE if col not in CATS]
    
    return {
        'BASE': BASE,
        'CATS': CATS,
        'NUMS': NUMS,
        'train': train.copy(),
        'test': test.copy()
    }


def print_feature_summary(feature_dict: dict):
    """
    特徴量のサマリーを表示
    
    Parameters:
    -----------
    feature_dict : dict
        get_base_features()の戻り値
    """
    print("=" * 60)
    print("特徴量の分類")
    print("=" * 60)
    print(f'  - ベース特徴量数: {len(feature_dict["BASE"])}')
    print(f'  - カテゴリ変数: {len(feature_dict["CATS"])} → {feature_dict["CATS"]}')
    print(f'  - 数値変数: {len(feature_dict["NUMS"])} → {feature_dict["NUMS"]}')
    print("=" * 60)
