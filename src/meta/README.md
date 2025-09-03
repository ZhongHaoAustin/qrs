# Meta模块说明

## 概述

Meta模块提供分析框架的核心元功能，作为整个分析系统的基础框架层。它定义了模式识别、特征构建和评估指标的标准接口和通用功能。

## 模块结构

```
src/meta/
├── __init__.py                    # 统一导出接口
├── example_usage.py               # 使用示例
├── README.md                      # 说明文档
├── pattern_recognition.py         # 模式识别框架
├── feature_construction.py        # 特征构建框架
└── evaluation_metrics.py          # 评估指标框架
```

## 核心功能

### 1. 模式识别框架 (Pattern Recognition Framework)

**文件**: `pattern_recognition.py`

**核心函数**:
- `identify_patterns()` - 模式识别主函数
- `register_pattern()` - 模式注册装饰器

**功能**:
- 提供模式识别的标准接口
- 支持多种模式识别算法的组合
- 使用装饰器模式便于扩展

### 2. 特征构建框架 (Feature Construction Framework)

**文件**: `feature_construction.py`

**核心函数**:
- `build_features()` - 特征构建主函数
- `register_feature()` - 特征注册装饰器
- `validate_feature_data()` - 数据验证函数
- `create_lag_features()` - 滞后特征创建
- `create_rolling_features()` - 滚动特征创建

**功能**:
- 提供特征工程的标准接口
- 支持多种特征构建算法的组合
- 提供常用的特征工程工具函数

### 3. 评估指标框架 (Evaluation Metrics Framework)

**文件**: `evaluation_metrics.py`

**核心函数**:
- `calculate_metrics()` - 指标计算主函数
- `register_metric()` - 指标注册装饰器
- `validate_metric_data()` - 数据验证函数
- `calculate_metric_summary()` - 指标汇总统计

**功能**:
- 提供评估指标的标准接口
- 支持多种指标计算算法的组合
- 提供指标汇总和统计功能

## 使用方式

### 1. 基本导入

```python
from src.meta import identify_patterns, build_features, calculate_metrics
```

### 2. 注册自定义函数

```python
from src.meta import register_pattern, register_feature, register_metric

@register_pattern("my_pattern")
def my_pattern_function(df):
    # 实现模式识别逻辑
    return df

@register_feature("my_feature")
def my_feature_function(df):
    # 实现特征构建逻辑
    return df

@register_metric("my_metric")
def my_metric_function(df):
    # 实现指标计算逻辑
    return 0.0
```

### 3. 完整分析流水线

```python
from src.meta import identify_patterns, build_features, calculate_metrics

# 模式识别
pattern_functions = [my_pattern_function]
df = identify_patterns(df, pattern_functions)

# 特征构建
feature_functions = [my_feature_function]
df = build_features(df, feature_functions)

# 指标计算
metric_functions = [my_metric_function]
metrics = calculate_metrics(df, metric_functions)
```

## 设计原则

1. **框架化设计**: 提供标准接口，便于扩展和组合
2. **装饰器模式**: 使用装饰器注册函数，简化使用
3. **函数式编程**: 支持函数组合，易于测试和维护
4. **数据验证**: 提供数据验证功能，确保数据质量
5. **工具函数**: 提供常用的工具函数，提高开发效率

## 扩展方式

### 添加新模式识别函数

```python
@register_pattern("new_pattern")
def identify_new_pattern(df: pd.DataFrame) -> pd.DataFrame:
    # 实现新的模式识别逻辑
    return df
```

### 添加新特征构建函数

```python
@register_feature("new_feature")
def construct_new_feature(df: pd.DataFrame) -> pd.DataFrame:
    # 实现新的特征构建逻辑
    return df
```

### 添加新评估指标函数

```python
@register_metric("new_metric")
def calculate_new_metric(df: pd.DataFrame) -> float:
    # 实现新的指标计算逻辑
    return 0.0
```

## 与Analysis模块的关系

- **Meta模块**: 提供核心框架和标准接口
- **Analysis模块**: 提供具体的实现和业务逻辑

Meta模块是Analysis模块的基础，Analysis模块中的具体实现都基于Meta模块定义的框架。

## 注意事项

1. 所有函数都应该遵循Meta模块定义的接口规范
2. 使用装饰器注册的函数会自动获得元数据
3. 数据验证函数可以帮助确保输入数据的质量
4. 工具函数可以简化常见的特征工程任务
