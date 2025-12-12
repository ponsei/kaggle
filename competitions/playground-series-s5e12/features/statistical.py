"""
統計的特徴量の作成
コレステロール比率、血圧関連、生活習慣スコアなど
"""
import pandas as pd
import numpy as np


def create_cholesterol_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    コレステロール関連の統計的特徴量を作成
    
    Parameters:
    -----------
    df : pd.DataFrame
        入力データフレーム
    
    Returns:
    --------
    pd.DataFrame : 特徴量を追加したデータフレーム
    """
    df = df.copy()
    
    # 非HDLコレステロール（総コレステロール - HDL）
    if 'cholesterol_total' in df.columns and 'hdl_cholesterol' in df.columns:
        df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    # LDL/HDL比
    if 'ldl_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
        df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-6)
    
    # 非HDL/HDL比
    if 'cholesterol_total' in df.columns and 'hdl_cholesterol' in df.columns:
        df['non_hdl_hdl_ratio'] = (df['cholesterol_total'] - df['hdl_cholesterol']) / (df['hdl_cholesterol'] + 1e-6)
    
    # 総コレステロール/HDL比
    if 'cholesterol_total' in df.columns and 'hdl_cholesterol' in df.columns:
        df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-6)
    
    return df


def create_blood_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    血圧関連の統計的特徴量を作成
    
    Parameters:
    -----------
    df : pd.DataFrame
        入力データフレーム
    
    Returns:
    --------
    pd.DataFrame : 特徴量を追加したデータフレーム
    """
    df = df.copy()
    
    # 脈圧（収縮期血圧 - 拡張期血圧）
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # 平均動脈圧（MAP）
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['systolic_bp'] - df['diastolic_bp']) / 3
    
    # 血圧比
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['systolic_diastolic_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1e-6)
    
    return df


def create_lifestyle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生活習慣関連の統計的特徴量を作成
    
    Parameters:
    -----------
    df : pd.DataFrame
        入力データフレーム
    
    Returns:
    --------
    pd.DataFrame : 特徴量を追加したデータフレーム
    """
    df = df.copy()
    
    # 活動スコア（WHO推奨値150分との比較）
    if 'physical_activity_minutes_per_week' in df.columns:
        df['activity_score'] = df['physical_activity_minutes_per_week'] / 150.0
    
    # 活動量のカテゴリ化
    if 'physical_activity_minutes_per_week' in df.columns:
        df['activity_level'] = pd.cut(
            df['physical_activity_minutes_per_week'],
            bins=[0, 75, 150, 300, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(float)
    
    # 睡眠スコア（推奨7-8時間との比較）
    if 'sleep_hours_per_day' in df.columns:
        df['sleep_score'] = df['sleep_hours_per_day'] / 7.5
    
    # 生活習慣スコア（複合指標）
    if all(col in df.columns for col in ['diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day']):
        df['lifestyle_score'] = (
            df['diet_score'] / 10.0 +  # 正規化
            (df['sleep_hours_per_day'] / 7.5) * 2 -  # 睡眠スコア
            (df['screen_time_hours_per_day'] / 8.0)  # スクリーンタイム（逆スコア）
        )
    
    return df


def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    年齢関連の特徴量を作成
    
    Parameters:
    -----------
    df : pd.DataFrame
        入力データフレーム
    
    Returns:
    --------
    pd.DataFrame : 特徴量を追加したデータフレーム
    """
    df = df.copy()
    
    if 'age' in df.columns:
        # 年齢層のカテゴリ化
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 30, 40, 50, 60, 100],
            labels=[1, 2, 3, 4, 5]
        ).astype(float)
        
        # 年齢の二乗
        df['age_squared'] = df['age'] ** 2
        
        # 年齢の平方根
        df['age_sqrt'] = np.sqrt(df['age'])
    
    return df


def create_bmi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    BMI関連の特徴量を作成
    
    Parameters:
    -----------
    df : pd.DataFrame
        入力データフレーム
    
    Returns:
    --------
    pd.DataFrame : 特徴量を追加したデータフレーム
    """
    df = df.copy()
    
    if 'bmi' in df.columns:
        # BMIのカテゴリ化（WHO基準）
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, float('inf')],
            labels=[0, 1, 2, 3]  # Underweight, Normal, Overweight, Obese
        ).astype(float)
        
        # BMIの二乗
        df['bmi_squared'] = df['bmi'] ** 2
        
        # BMIの対数変換
        df['bmi_log'] = np.log1p(df['bmi'])
    
    return df


def create_all_statistical_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    すべての統計的特徴量を作成
    
    Parameters:
    -----------
    train : pd.DataFrame
        訓練データ
    test : pd.DataFrame
        テストデータ
    
    Returns:
    --------
    tuple : (train, test) 特徴量を追加したデータフレーム
    """
    train = train.copy()
    test = test.copy()
    
    # 各特徴量グループを適用
    train = create_cholesterol_features(train)
    test = create_cholesterol_features(test)
    
    train = create_blood_pressure_features(train)
    test = create_blood_pressure_features(test)
    
    train = create_lifestyle_features(train)
    test = create_lifestyle_features(test)
    
    train = create_age_features(train)
    test = create_age_features(test)
    
    train = create_bmi_features(train)
    test = create_bmi_features(test)
    
    return train, test
