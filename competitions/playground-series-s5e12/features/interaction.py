"""
相互作用特徴量の作成
高重要度特徴量同士の相互作用、低重要度特徴量の組み合わせなど
"""
import pandas as pd
import numpy as np


def create_high_importance_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    高重要度特徴量の相互作用特徴量を作成
    
    Parameters:
    -----------
    df : pd.DataFrame
        入力データフレーム
    
    Returns:
    --------
    pd.DataFrame : 特徴量を追加したデータフレーム
    """
    df = df.copy()
    
    # physical_activity_minutes_per_week との相互作用
    if 'physical_activity_minutes_per_week' in df.columns:
        if 'age' in df.columns:
            df['age_activity_ratio'] = df['age'] / (df['physical_activity_minutes_per_week'] + 1)
            df['age_activity_interaction'] = df['age'] * df['physical_activity_minutes_per_week']
        
        if 'bmi' in df.columns:
            df['bmi_activity_interaction'] = df['bmi'] * df['physical_activity_minutes_per_week']
            df['bmi_activity_ratio'] = df['bmi'] / (df['physical_activity_minutes_per_week'] + 1)
        
        if 'family_history_diabetes' in df.columns:
            df['family_history_activity'] = df['family_history_diabetes'] * df['physical_activity_minutes_per_week']
    
    # family_history_diabetes との相互作用
    if 'family_history_diabetes' in df.columns:
        if 'age' in df.columns:
            df['family_history_age'] = df['family_history_diabetes'] * df['age']
        
        if 'bmi' in df.columns:
            df['family_history_bmi'] = df['family_history_diabetes'] * df['bmi']
        
        if 'triglycerides' in df.columns:
            df['family_history_triglycerides'] = df['family_history_diabetes'] * df['triglycerides']
    
    # age との相互作用
    if 'age' in df.columns:
        if 'bmi' in df.columns:
            df['age_bmi_interaction'] = df['age'] * df['bmi']
        
        if 'triglycerides' in df.columns:
            df['age_triglycerides_interaction'] = df['age'] * df['triglycerides']
        
        if 'ldl_cholesterol' in df.columns:
            df['age_ldl_interaction'] = df['age'] * df['ldl_cholesterol']
    
    return df


def create_cholesterol_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    コレステロール関連の相互作用特徴量を作成
    
    Parameters:
    -----------
    df : pd.DataFrame
        入力データフレーム
    
    Returns:
    --------
    pd.DataFrame : 特徴量を追加したデータフレーム
    """
    df = df.copy()
    
    # コレステロール値同士の相互作用
    if all(col in df.columns for col in ['cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol']):
        df['total_hdl_ldl_interaction'] = df['cholesterol_total'] * df['hdl_cholesterol'] * df['ldl_cholesterol']
    
    if 'ldl_cholesterol' in df.columns and 'triglycerides' in df.columns:
        df['ldl_triglycerides_interaction'] = df['ldl_cholesterol'] * df['triglycerides']
    
    return df


def create_lifestyle_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    生活習慣関連の相互作用特徴量を作成
    
    Parameters:
    -----------
    df : pd.DataFrame
        入力データフレーム
    
    Returns:
    --------
    pd.DataFrame : 特徴量を追加したデータフレーム
    """
    df = df.copy()
    
    # 食事と運動
    if 'diet_score' in df.columns and 'physical_activity_minutes_per_week' in df.columns:
        df['diet_activity_interaction'] = df['diet_score'] * df['physical_activity_minutes_per_week']
    
    # 睡眠とスクリーンタイム
    if 'sleep_hours_per_day' in df.columns and 'screen_time_hours_per_day' in df.columns:
        df['sleep_screen_interaction'] = df['sleep_hours_per_day'] * df['screen_time_hours_per_day']
        df['sleep_screen_ratio'] = df['sleep_hours_per_day'] / (df['screen_time_hours_per_day'] + 1)
    
    # アルコールと喫煙
    if 'alcohol_consumption_per_week' in df.columns and 'smoking_status' in df.columns:
        df['alcohol_smoking_interaction'] = df['alcohol_consumption_per_week'] * df['smoking_status']
    
    return df


def create_demographic_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    人口統計学的特徴量の相互作用を作成（低重要度特徴量の活用）
    
    Parameters:
    -----------
    df : pd.DataFrame
        入力データフレーム
    
    Returns:
    --------
    pd.DataFrame : 特徴量を追加したデータフレーム
    """
    df = df.copy()
    
    # 教育と収入
    if 'education_level' in df.columns and 'income_level' in df.columns:
        df['education_income_interaction'] = df['education_level'] * df['income_level']
    
    # 民族と収入
    if 'ethnicity' in df.columns and 'income_level' in df.columns:
        df['ethnicity_income_interaction'] = df['ethnicity'] * df['income_level']
    
    # 雇用と収入
    if 'employment_status' in df.columns and 'income_level' in df.columns:
        df['employment_income_interaction'] = df['employment_status'] * df['income_level']
    
    return df


def create_all_interaction_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    すべての相互作用特徴量を作成
    
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
    
    # 各相互作用グループを適用
    train = create_high_importance_interactions(train)
    test = create_high_importance_interactions(test)
    
    train = create_cholesterol_interactions(train)
    test = create_cholesterol_interactions(test)
    
    train = create_lifestyle_interactions(train)
    test = create_lifestyle_interactions(test)
    
    train = create_demographic_interactions(train)
    test = create_demographic_interactions(test)
    
    return train, test
