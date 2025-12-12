"""
特徴量選択関連の処理
特徴量重要度に基づく選択、相関分析など
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def get_feature_importance_from_models(models: List, feature_names: List[str], 
                                      importance_type: str = 'gain') -> pd.DataFrame:
    """
    複数のLightGBMモデルから特徴量重要度の平均を計算
    
    Parameters:
    -----------
    models : List
        LightGBMモデルのリスト
    feature_names : List[str]
        特徴量名のリスト
    importance_type : str
        重要度のタイプ（'gain', 'split', 'gain'）
    
    Returns:
    --------
    pd.DataFrame : 特徴量名と重要度のデータフレーム
    """
    feature_importance_dict = {feat: 0.0 for feat in feature_names}
    
    for model in models:
        importances = model.feature_importance(importance_type=importance_type)
        for feat, imp in zip(feature_names, importances):
            feature_importance_dict[feat] += imp / len(models)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': [feature_importance_dict[feat] for feat in feature_names]
    }).sort_values('importance', ascending=False)
    
    return feature_importance


def select_features_by_importance(feature_importance: pd.DataFrame, 
                                 n_features: int = None,
                                 threshold: float = None) -> List[str]:
    """
    特徴量重要度に基づいて特徴量を選択
    
    Parameters:
    -----------
    feature_importance : pd.DataFrame
        特徴量重要度のデータフレーム（'feature', 'importance'カラム）
    n_features : int, optional
        選択する特徴量数（上位n_features個）
    threshold : float, optional
        重要度の閾値（この値以上の特徴量を選択）
    
    Returns:
    --------
    List[str] : 選択された特徴量名のリスト
    """
    if n_features is not None:
        selected = feature_importance.head(n_features)['feature'].tolist()
    elif threshold is not None:
        selected = feature_importance[feature_importance['importance'] >= threshold]['feature'].tolist()
    else:
        # デフォルト: 重要度が0より大きい特徴量を選択
        selected = feature_importance[feature_importance['importance'] > 0]['feature'].tolist()
    
    return selected


def remove_highly_correlated_features(df: pd.DataFrame, 
                                     target_col: str = None,
                                     threshold: float = 0.95) -> List[str]:
    """
    相関の高い特徴量を削除（多重共線性の回避）
    
    Parameters:
    -----------
    df : pd.DataFrame
        データフレーム
    target_col : str, optional
        目的変数のカラム名（除外する）
    threshold : float
        相関係数の閾値（この値以上で相関が高いと判断）
    
    Returns:
    --------
    List[str] : 残すべき特徴量名のリスト
    """
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])
    
    # 数値特徴量のみを対象
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols]
    
    # 相関行列を計算
    corr_matrix = df_numeric.corr().abs()
    
    # 上三角行列を取得（対称行列なので）
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # 相関の高い特徴量ペアを検出
    to_remove = []
    for col in upper_triangle.columns:
        if col in to_remove:
            continue
        high_corr = upper_triangle.index[upper_triangle[col] >= threshold].tolist()
        to_remove.extend(high_corr)
    
    # 残すべき特徴量
    to_keep = [col for col in numeric_cols if col not in to_remove]
    
    return to_keep


def select_features_combined(train: pd.DataFrame,
                             models: List,
                             feature_names: List[str],
                             target_col: str = None,
                             n_features: int = None,
                             importance_threshold: float = None,
                             correlation_threshold: float = 0.95) -> Tuple[List[str], pd.DataFrame]:
    """
    特徴量重要度と相関分析を組み合わせて特徴量を選択
    
    Parameters:
    -----------
    train : pd.DataFrame
        訓練データ
    models : List
        LightGBMモデルのリスト
    feature_names : List[str]
        特徴量名のリスト
    target_col : str, optional
        目的変数のカラム名
    n_features : int, optional
        選択する特徴量数
    importance_threshold : float, optional
        重要度の閾値
    correlation_threshold : float
        相関係数の閾値
    
    Returns:
    --------
    Tuple[List[str], pd.DataFrame] : (選択された特徴量リスト, 特徴量重要度データフレーム)
    """
    # 1. 特徴量重要度を計算
    feature_importance = get_feature_importance_from_models(models, feature_names)
    
    # 2. 重要度に基づいて選択
    if n_features is not None:
        selected_by_importance = select_features_by_importance(
            feature_importance, n_features=n_features
        )
    elif importance_threshold is not None:
        selected_by_importance = select_features_by_importance(
            feature_importance, threshold=importance_threshold
        )
    else:
        # 重要度が0より大きい特徴量
        selected_by_importance = select_features_by_importance(feature_importance)
    
    # 3. 相関の高い特徴量を削除
    train_selected = train[selected_by_importance]
    selected_final = remove_highly_correlated_features(
        train_selected, target_col=target_col, threshold=correlation_threshold
    )
    
    return selected_final, feature_importance


def compare_feature_sets(train: pd.DataFrame,
                         y_train: pd.Series,
                         feature_sets: Dict[str, List[str]],
                         cv_func,
                         **cv_kwargs) -> pd.DataFrame:
    """
    複数の特徴量セットの性能を比較
    
    Parameters:
    -----------
    train : pd.DataFrame
        訓練データ
    y_train : pd.Series
        目的変数
    feature_sets : Dict[str, List[str]]
        特徴量セットの辞書（例: {'base': [...], 'with_stats': [...]}）
    cv_func : callable
        交差検証関数
    **cv_kwargs : dict
        交差検証関数に渡す引数
    
    Returns:
    --------
    pd.DataFrame : 各特徴量セットの性能比較結果
    """
    results = []
    
    for set_name, features in feature_sets.items():
        X = train[features]
        cv_scores = cv_func(X, y_train, **cv_kwargs)
        
        results.append({
            'feature_set': set_name,
            'n_features': len(features),
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'features': features
        })
    
    return pd.DataFrame(results)
