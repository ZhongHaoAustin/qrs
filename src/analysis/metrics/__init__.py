"""
评估指标模块 - Evaluation Metrics Module

提供模型和策略的评估指标功能，包括：
- 预测准确率指标
- 收益风险指标
- 模式识别效果指标
- 特征重要性指标
"""

from src.analysis.metrics.evaluation_metrics import calculate_metrics, register_metric
from src.analysis.metrics.pattern_metrics import calculate_pattern_effectiveness
from src.analysis.metrics.performance_metrics import calculate_performance_metrics
from src.analysis.metrics.prediction_metrics import calculate_prediction_accuracy

__all__ = [
    "calculate_metrics",
    "register_metric",
    "calculate_prediction_accuracy",
    "calculate_performance_metrics",
    "calculate_pattern_effectiveness",
]
