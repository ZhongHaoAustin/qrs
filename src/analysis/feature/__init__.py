"""
特征构建模块 - Feature Construction Module

提供订单簿数据的特征工程功能，包括：
- 订单流特征构建
- 技术指标特征
- 统计特征
- 时间序列特征
"""

from src.analysis.feature.feature_construction import build_features, register_feature
from src.analysis.feature.order_flow_features import construct_order_flow_features
from src.analysis.feature.statistical_features import construct_statistical_features
from src.analysis.feature.technical_features import construct_technical_features

__all__ = [
    "build_features",
    "register_feature",
    "construct_order_flow_features",
    "construct_technical_features",
    "construct_statistical_features",
]
