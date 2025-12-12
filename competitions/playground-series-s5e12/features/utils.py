"""
特徴量エンジニアリングのユーティリティ関数
段階的な特徴量追加、実験管理など
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
import sys
import os

# 親ディレクトリのfeaturesモジュールをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_features_phase1(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 1: 高重要度特徴量の相互作用特徴量を作成
    
    Parameters:
    -----------
    train : pd.DataFrame
        訓練データ
    test : pd.DataFrame
        テストデータ
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame] : (train, test)
    """
    from .interaction import create_high_importance_interactions
    
    train = create_high_importance_interactions(train)
    test = create_high_importance_interactions(test)
    
    return train, test


def create_features_phase2(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 2: 統計的特徴量を作成
    
    Parameters:
    -----------
    train : pd.DataFrame
        訓練データ
    test : pd.DataFrame
        テストデータ
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame] : (train, test)
    """
    from .statistical import create_all_statistical_features
    
    train, test = create_all_statistical_features(train, test)
    
    return train, test


def create_features_phase3(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 3: その他の相互作用特徴量を作成
    
    Parameters:
    -----------
    train : pd.DataFrame
        訓練データ
    test : pd.DataFrame
        テストデータ
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame] : (train, test)
    """
    from .interaction import (
        create_cholesterol_interactions,
        create_lifestyle_interactions,
        create_demographic_interactions
    )
    
    train = create_cholesterol_interactions(train)
    test = create_cholesterol_interactions(test)
    
    train = create_lifestyle_interactions(train)
    test = create_lifestyle_interactions(test)
    
    train = create_demographic_interactions(train)
    test = create_demographic_interactions(test)
    
    return train, test


def create_features_incremental(train: pd.DataFrame, 
                                test: pd.DataFrame,
                                phases: List[int] = [1, 2, 3]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    段階的に特徴量を追加
    
    Parameters:
    -----------
    train : pd.DataFrame
        訓練データ
    test : pd.DataFrame
        テストデータ
    phases : List[int]
        実行するフェーズのリスト（1, 2, 3）
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, Dict] : (train, test, feature_info)
        feature_info: 各フェーズで追加された特徴量の情報
    """
    train = train.copy()
    test = test.copy()
    feature_info = {
        'initial_features': len([c for c in train.columns if c not in ['id', 'diagnosed_diabetes']]),
        'phases': {}
    }
    
    phase_functions = {
        1: create_features_phase1,
        2: create_features_phase2,
        3: create_features_phase3
    }
    
    for phase in phases:
        if phase not in phase_functions:
            continue
        
        n_before = len([c for c in train.columns if c not in ['id', 'diagnosed_diabetes']])
        
        train, test = phase_functions[phase](train, test)
        
        n_after = len([c for c in train.columns if c not in ['id', 'diagnosed_diabetes']])
        n_added = n_after - n_before
        
        feature_info['phases'][f'phase_{phase}'] = {
            'features_before': n_before,
            'features_after': n_after,
            'features_added': n_added
        }
    
    feature_info['total_features'] = len([c for c in train.columns if c not in ['id', 'diagnosed_diabetes']])
    
    return train, test, feature_info


def get_feature_list_by_phase(train: pd.DataFrame,
                              base_features: List[str],
                              phases: List[int] = [1, 2, 3]) -> Dict[str, List[str]]:
    """
    各フェーズで追加された特徴量のリストを取得
    
    Parameters:
    -----------
    train : pd.DataFrame
        特徴量追加後の訓練データ
    base_features : List[str]
        ベース特徴量のリスト
    phases : List[int]
        実行したフェーズのリスト
    
    Returns:
    --------
    Dict[str, List[str]] : 各フェーズの特徴量リスト
    """
    feature_sets = {
        'base': base_features.copy()
    }
    
    # Phase 1の特徴量（高重要度相互作用）
    if 1 in phases:
        phase1_features = [c for c in train.columns 
                          if c not in base_features 
                          and c not in ['id', 'diagnosed_diabetes']
                          and any(keyword in c for keyword in [
                              'activity', 'family_history', 'age_', 'bmi_'
                          ])]
        feature_sets['phase1'] = base_features + phase1_features
    
    # Phase 2の特徴量（統計的特徴量）
    if 2 in phases:
        phase2_features = [c for c in train.columns 
                          if c not in feature_sets.get('phase1', base_features)
                          and c not in ['id', 'diagnosed_diabetes']
                          and any(keyword in c for keyword in [
                              'cholesterol', 'pressure', 'score', 'group', 'squared', 'log', 'sqrt'
                          ])]
        prev_features = feature_sets.get('phase1', base_features)
        feature_sets['phase2'] = prev_features + phase2_features
    
    # Phase 3の特徴量（その他の相互作用）
    if 3 in phases:
        phase3_features = [c for c in train.columns 
                          if c not in feature_sets.get('phase2', base_features)
                          and c not in ['id', 'diagnosed_diabetes']]
        prev_features = feature_sets.get('phase2', base_features)
        feature_sets['phase3'] = prev_features + phase3_features
    
    return feature_sets


def print_feature_summary_by_phase(feature_info: Dict):
    """
    フェーズごとの特徴量サマリーを表示
    
    Parameters:
    -----------
    feature_info : Dict
        create_features_incremental()の戻り値のfeature_info
    """
    print("=" * 60)
    print("特徴量追加サマリー")
    print("=" * 60)
    print(f"初期特徴量数: {feature_info['initial_features']}")
    print()
    
    for phase_name, phase_info in feature_info['phases'].items():
        print(f"{phase_name.upper()}:")
        print(f"  追加前: {phase_info['features_before']}個")
        print(f"  追加後: {phase_info['features_after']}個")
        print(f"  追加数: {phase_info['features_added']}個")
        print()
    
    print(f"合計特徴量数: {feature_info['total_features']}個")
    print("=" * 60)
