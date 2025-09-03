# Analysis模块重构说明

## 概述

根据ARCHITECTURE.md中的要求，我们重新设计了analysis目录结构，按照pattern、feature、metrics三个模块进行划分，每个模块提供特定类型的函数供其他地方调用。

## 新的目录结构

```
src/analysis/
├── __init__.py                    # 统一导出所有模块的公共接口
├── pattern_recognition_research.py # 主研究文件（使用FP风格）
├── example_usage.py               # 使用示例
├── order_flow_np.py              # 订单流分析（保留原有功能）
├── pattern/                      # 模式识别模块
│   ├── __init__.py
│   ├── pattern_recognition.py    # 模式识别核心框架
│   ├── order_imbalance.py        # 订单不平衡模式识别
│   └── order_flow_patterns.py    # 订单流模式识别
├── feature/                      # 特征构建模块
│   ├── __init__.py
│   ├── feature_construction.py   # 特征构建核心框架
│   ├── order_flow_features.py    # 订单流特征构建
│   ├── technical_features.py     # 技术指标特征构建
│   └── statistical_features.py   # 统计特征构建
└── metrics/                      # 评估指标模块
    ├── __init__.py
    ├── evaluation_metrics.py     # 评估指标核心框架
    ├── prediction_metrics.py     # 预测指标
    ├── performance_metrics.py    # 性能指标
    └── pattern_metrics.py        # 模式效果指标
```

## 模块功能说明

### 1. Pattern模块 - 模式识别

**核心功能：**
- 识别订单簿中的各种模式
- 提供模式注册和识别框架
- 支持多种模式识别算法

**主要函数：**
- `identify_patterns()` - 模式识别主函数
- `identify_large_order_imbalance()` - 大单不平衡模式识别
- `identify_order_flow_patterns()` - 订单流模式识别
- `register_pattern()` - 模式注册装饰器

### 2. Feature模块 - 特征构建

**核心功能：**
- 基于订单簿数据构建各种特征
- 提供特征工程框架
- 支持技术指标、统计特征等

**主要函数：**
- `build_features()` - 特征构建主函数
- `construct_order_flow_features()` - 订单流特征构建
- `construct_technical_features()` - 技术指标特征构建
- `construct_statistical_features()` - 统计特征构建
- `register_feature()` - 特征注册装饰器

### 3. Metrics模块 - 评估指标

**核心功能：**
- 计算模型和策略的评估指标
- 提供指标计算框架
- 支持预测、性能、模式效果等指标

**主要函数：**
- `calculate_metrics()` - 指标计算主函数
- `calculate_prediction_accuracy()` - 预测准确率计算
- `calculate_performance_metrics()` - 性能指标计算
- `calculate_pattern_effectiveness()` - 模式效果指标计算
- `register_metric()` - 指标注册装饰器

## 使用方式

### 1. 导入模块

```python
# 导入整个模块
from src.analysis import (
    identify_patterns,
    build_features,
    calculate_metrics
)

# 或者导入特定子模块
from src.analysis.pattern import identify_large_order_imbalance
from src.analysis.feature import construct_order_flow_features
from src.analysis.metrics import calculate_prediction_accuracy
```

### 2. 使用示例

```python
# 模式识别
pattern_functions = [identify_large_order_imbalance, identify_order_flow_patterns]
df = identify_patterns(df, pattern_functions)

# 特征构建
feature_functions = [construct_order_flow_features, construct_technical_features]
df = build_features(df, feature_functions)

# 指标计算
metric_functions = [calculate_prediction_accuracy, calculate_performance_metrics]
metrics = calculate_metrics(df, metric_functions)
```

### 3. 完整流水线

```python
from src.analysis.pattern_recognition_research import main

# 使用主研究函数（FP风格）
result = main("config/orderbook_config.yaml")
```

## 设计原则

1. **模块化设计**：每个模块负责特定功能，职责清晰
2. **函数式编程**：main函数由多个函数组合而成，易于测试和维护
3. **绝对导入**：遵循ARCHITECTURE.md中的导入规范
4. **装饰器模式**：使用装饰器注册函数，便于扩展
5. **统一接口**：通过__init__.py提供统一的公共接口

## 扩展方式

### 添加新模式识别函数

```python
from src.analysis.pattern.pattern_recognition import register_pattern

@register_pattern("new_pattern")
def identify_new_pattern(df: pd.DataFrame) -> pd.DataFrame:
    # 实现新的模式识别逻辑
    return df
```

### 添加新特征构建函数

```python
from src.analysis.feature.feature_construction import register_feature

@register_feature("new_feature")
def construct_new_feature(df: pd.DataFrame) -> pd.DataFrame:
    # 实现新的特征构建逻辑
    return df
```

### 添加新评估指标函数

```python
from src.analysis.metrics.evaluation_metrics import register_metric

@register_metric("new_metric")
def calculate_new_metric(df: pd.DataFrame) -> float:
    # 实现新的指标计算逻辑
    return 0.0
```

## 注意事项

1. 所有导入都使用绝对路径，符合项目规范
2. 函数命名遵循统一的命名约定
3. 每个函数都有完整的文档字符串
4. 使用loguru进行日志记录
5. 保持向后兼容性，原有功能继续可用
