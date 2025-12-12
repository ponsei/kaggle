"""
特徴量エンジニアリングモジュール

使用方法:
    from features import (
        get_base_features, create_all_statistical_features,
        create_all_interaction_features, label_encode_categorical
    )
    
    # ベース特徴量の取得
    feature_dict = get_base_features(train, test)
    
    # 統計的特徴量の作成
    train, test = create_all_statistical_features(train, test)
    
    # 相互作用特徴量の作成
    train, test = create_all_interaction_features(train, test)
    
    # エンコーディング
    train, test, encoders = label_encode_categorical(train, test, categorical_cols)
"""

from .base import get_base_features, print_feature_summary
from .statistical import (
    create_cholesterol_features,
    create_blood_pressure_features,
    create_lifestyle_features,
    create_age_features,
    create_bmi_features,
    create_all_statistical_features
)
from .interaction import (
    create_high_importance_interactions,
    create_cholesterol_interactions,
    create_lifestyle_interactions,
    create_demographic_interactions,
    create_all_interaction_features
)
from .encoding import (
    label_encode_categorical,
    ordinal_encode,
    target_encode,
    frequency_encode
)
from .selection import (
    get_feature_importance_from_models,
    select_features_by_importance,
    remove_highly_correlated_features,
    select_features_combined,
    compare_feature_sets
)
from .utils import (
    create_features_phase1,
    create_features_phase2,
    create_features_phase3,
    create_features_incremental,
    get_feature_list_by_phase,
    print_feature_summary_by_phase
)

__all__ = [
    # base
    'get_base_features',
    'print_feature_summary',
    # statistical
    'create_cholesterol_features',
    'create_blood_pressure_features',
    'create_lifestyle_features',
    'create_age_features',
    'create_bmi_features',
    'create_all_statistical_features',
    # interaction
    'create_high_importance_interactions',
    'create_cholesterol_interactions',
    'create_lifestyle_interactions',
    'create_demographic_interactions',
    'create_all_interaction_features',
    # encoding
    'label_encode_categorical',
    'ordinal_encode',
    'target_encode',
    'frequency_encode',
    # selection
    'get_feature_importance_from_models',
    'select_features_by_importance',
    'remove_highly_correlated_features',
    'select_features_combined',
    'compare_feature_sets',
    # utils
    'create_features_phase1',
    'create_features_phase2',
    'create_features_phase3',
    'create_features_incremental',
    'get_feature_list_by_phase',
    'print_feature_summary_by_phase',
]
