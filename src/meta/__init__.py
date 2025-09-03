"""
元分析模块 - Meta Analysis Module

提供分析框架的核心元功能，包括：
- 模式识别框架
- 特征构建框架
- 评估指标框架
- 元学习功能
"""

from src.meta.evaluation_metrics import (
    calculate_metric_summary,
    calculate_metrics,
    register_metric,
    validate_metric_data,
)
from src.meta.feature_construction import (
    build_features,
    create_lag_features,
    create_rolling_features,
    register_feature,
    validate_feature_data,
)
from src.meta.pattern_recognition import (
    identify_patterns,
    register_pattern,
)

__all__ = [
    # 模式识别框架
    "identify_patterns",
    "register_pattern",
    # 特征构建框架
    "build_features",
    "register_feature",
    "validate_feature_data",
    "create_lag_features",
    "create_rolling_features",
    # 评估指标框架
    "calculate_metrics",
    "register_metric",
    "validate_metric_data",
    "calculate_metric_summary",
]
