"""
预测指标模块 - Prediction Metrics Module

提供预测模型相关的评估指标。
"""

from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.meta import register_metric


@register_metric("prediction_accuracy")
def calculate_prediction_accuracy(df: pd.DataFrame) -> float:
    """
    计算模式预测的准确率。

    Args:
        df: 包含预测结果和实际结果的DataFrame

    Returns:
        预测准确率
    """
    if "large_order_imbalance" in df.columns and "price_movement" in df.columns:
        # 对于这个例子，我们假设价格运动是下一个tick的方向
        df["price_movement"] = (
            df["last_price"].diff().shift(-1).apply(lambda x: 1 if x > 0 else 0)
        )
        df["prediction"] = df["large_order_imbalance"].apply(lambda x: 1 if x else 0)

        accuracy = (df["prediction"] == df["price_movement"]).mean()
        return accuracy
    return 0.0


@register_metric("precision_recall_f1")
def calculate_precision_recall_f1(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算精确率、召回率和F1分数。

    Args:
        df: 包含预测结果和实际结果的DataFrame

    Returns:
        包含精确率、召回率和F1分数的字典
    """
    if "prediction" in df.columns and "price_movement" in df.columns:
        # 移除NaN值
        valid_mask = ~(df["prediction"].isna() | df["price_movement"].isna())
        y_true = df["price_movement"][valid_mask]
        y_pred = df["prediction"][valid_mask]

        if len(y_true) > 0 and len(y_pred) > 0:
            precision = precision_score(
                y_true, y_pred, average="binary", zero_division=0
            )
            recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

            return {"precision": precision, "recall": recall, "f1_score": f1}

    return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}


@register_metric("confusion_matrix_metrics")
def calculate_confusion_matrix_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算混淆矩阵相关指标。

    Args:
        df: 包含预测结果和实际结果的DataFrame

    Returns:
        包含混淆矩阵指标的字典
    """
    if "prediction" in df.columns and "price_movement" in df.columns:
        # 移除NaN值
        valid_mask = ~(df["prediction"].isna() | df["price_movement"].isna())
        y_true = df["price_movement"][valid_mask]
        y_pred = df["prediction"][valid_mask]

        if len(y_true) > 0 and len(y_pred) > 0:
            # 计算混淆矩阵
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()

            # 计算各种指标
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0
            negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0

            return {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "positive_predictive_value": positive_predictive_value,
                "negative_predictive_value": negative_predictive_value,
            }

    return {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "positive_predictive_value": 0.0,
        "negative_predictive_value": 0.0,
    }


@register_metric("prediction_timing")
def calculate_prediction_timing(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算预测时机相关指标。

    Args:
        df: 包含预测结果和时间数据的DataFrame

    Returns:
        包含预测时机指标的字典
    """
    if "large_order_imbalance" in df.columns and "datetime" in df.columns:
        # 计算模式识别的时间分布
        imbalance_events = df[df["large_order_imbalance"] == True]

        if len(imbalance_events) > 0:
            # 计算事件间隔
            time_diffs = imbalance_events["datetime"].diff().dt.total_seconds()
            time_diffs = time_diffs.dropna()

            if len(time_diffs) > 0:
                return {
                    "total_imbalance_events": len(imbalance_events),
                    "avg_event_interval": time_diffs.mean(),
                    "min_event_interval": time_diffs.min(),
                    "max_event_interval": time_diffs.max(),
                    "event_frequency": len(imbalance_events) / len(df),
                }

    return {
        "total_imbalance_events": 0,
        "avg_event_interval": 0.0,
        "min_event_interval": 0.0,
        "max_event_interval": 0.0,
        "event_frequency": 0.0,
    }
