# Orderbook数据分析框架架构文档

## 项目概述

本项目是一个基于订单簿数据的研究框架，旨在通过模式识别、特征构建和评价指标等模块，对金融市场中的订单簿数据进行深入分析。项目采用模块化设计，便于功能扩展和复用。

量化策略可以从模式识别的角度思考，比如基于orderbook数据把某种情况筛选出来，基于该情况之前的数据构建特征，基于该情况之后的数据构建评价指标，一步步去识别模式和细化特征，提升评估指标。请你使用FP编程风格（某个大功能的main函数接受配置文件，main函数由多个函数compose起来），设计三个抽象的模块，1. 模式识别模块，2. 特征构建模块，3. 评价指标模块。并且基于现有的项目框架完成抽象模块的构造并且给出一个模式识别的研究实例。

### 核心目标

1. **模式识别**：基于订单簿数据识别特定的市场模式
2. **特征构建**：基于识别出的模式构建有效的特征
3. **评价指标**：建立评估模型性能的指标体系

### 数据说明

订单簿数据相关详细说明请参考 [订单簿数据说明文档](../data_schema/orderbook.md)

## 系统架构

项目采用模块化设计，主要包含以下核心模块：

### 1. 数据模块 (Data Module)
负责数据的获取、预处理和验证：
- **数据获取**：从数据源获取订单簿数据
- **数据预处理**：清洗和格式化数据
- **数据验证**：确保数据质量和一致性

### 2. 分析模块 (Analysis Module)
负责核心的数据分析和计算：
- **订单流分析**：计算订单簿变化量(delta)
- **特征工程**：构建分析所需的特征
- **模式识别**：识别特定的市场模式

### 3. 可视化模块 (Visualization Module)
负责数据和分析结果的可视化展示：
- **图表绘制**：使用Bokeh绘制交互式图表
- **模式标记**：在图表上标记识别出的模式

### 4. 工具模块 (Utils Module)
提供通用的工具函数：
- **配置管理**：处理YAML配置文件
- **日志管理**：统一的日志输出
- **数据工具**：通用的数据处理函数

## 配置管理

配置文件使用YAML格式，包含必要的注释说明。配置文件存放在 [config](../config/) 目录下。

示例配置文件结构：
```yaml
# 数据获取参数
data:
  instrument_id: "SSE.10009482"  # 合约代码
  exchange: "SSE"               # 交易所
  start_date: 20250801          # 开始日期 (YYYYMMDD格式)
  end_date: 20250801            # 结束日期 (YYYYMMDD格式)

# 分析参数
analysis:
  window_size: 10               # 计算窗口大小
  enable_order_flow: true       # 是否启用订单流分析

# 输出设置
output:
  save_plots: true              # 是否保存图表
  plot_directory: "output"      # 图表保存目录
```

## 模块详细说明

### 数据模块 (src/data/)

#### data_fetching.py
提供数据获取功能：
- `fetch_tick_data()`: 获取指定日期的tick数据
- `fetch_tick_data_range()`: 获取日期范围内的tick数据

#### data_processing.py
提供数据处理功能：
- 数据清洗和格式化
- 数据验证和质量检查

### 分析模块 (src/analysis/)

#### order_flow_np.py
提供订单簿变化量(delta)计算功能：
- `calculate_order_book_delta_numpy()`: 使用numpy优化的订单簿变化量计算
- `calculate_order_book_delta_numpy_v2()`: 另一个优化版本

### 可视化模块 (src/visualization/)

#### plotting.py
提供数据可视化功能：
- 创建订单簿价格图表
- 创建成交量图表
- 创建订单簿变化量图表
- 添加模式标记功能

### 工具模块 (src/utils/)

#### config_manager.py
配置文件管理工具：
- 加载和解析YAML配置文件

#### logging_config.py
日志配置工具：
- `setup_logger()`: 设置模块日志记录器

#### data_utils.py
数据处理工具：
- `display_dataframe()`: 格式化显示DataFrame
- `validate_orderbook_data()`: 验证订单簿数据结构

## 代码规范

### 导入规范
所有代码文件必须使用绝对导入而不是相对导入，以确保清晰性并避免潜在的导入问题：

```python
# 正确 - 绝对导入
from src.analysis.order_flow import calculate_order_flow
from utils.logging_config import setup_logger

# 错误 - 相对导入
from .order_flow import calculate_order_flow
```

### 日志规范
所有模块必须使用loguru进行结构化日志输出，所有日志消息必须使用loguru的内置格式化选项进行格式化，并且写入日志文件的模式必须设置为"w"：

```python
from loguru import logger
from utils.logging_config import setup_logger

# 设置日志
logger = setup_logger("module_name")

# 使用示例
logger.info("开始执行分析")
logger.debug("正在处理数据行: {}", row_index)
logger.success("分析成功完成")
logger.error("发生错误: {}", error_message)
```

### DataFrame显示规范
在代码中，所有DataFrame输出必须使用`pandas`的`to_string()`方法进行格式化，以确保输出清晰且易于阅读，输出前用`\n`换行符：

```python
from utils.data_utils import display_dataframe

# 使用工具函数显示DataFrame
display_dataframe(df, title="分析结果", max_rows=10)

# 或者直接使用
print("\n" + df.to_string())
```

### 绘图规范
1. 使用bokeh进行绘图，绘图结果保存为html保存到`output`的合适目录下
2. 图片大小使用响应式布局，能够根据网页大小进行自适应调整
3. 图标的颜色选用与白色背景对比强烈的鲜艳颜色，参考 #1f77b4, #e377c2, #2ca02c, #d62728, #9467bd 等

## 性能优化

### 数据处理优化
```python
import numpy as np
import pandas as pd

def optimized_analysis(df: pd.DataFrame):
    """优化的分析函数"""
    # 使用numpy进行数值计算
    volume_array = df['volume'].values
    amount_array = df['amount'].values
    
    # 向量化计算
    vwap = np.divide(amount_array, volume_array, out=np.zeros_like(amount_array), where=volume_array!=0)
    
    return vwap
```

## 文档规范

### 函数文档
```python
def calculate_order_flow(df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
    """
    计算订单流指标
    
    Args:
        df (pd.DataFrame): 包含tick数据的DataFrame
        window_size (int): 计算窗口大小，默认10
    
    Returns:
        pd.DataFrame: 包含订单流指标的DataFrame
    
    Raises:
        ValueError: 当输入数据格式不正确时
        KeyError: 当缺少必需的列时
    
    Example:
        >>> df = load_data("data.csv")
        >>> result = calculate_order_flow(df, window_size=20)
        >>> print(result.head())
    """
    pass
```

## 开发流程

### 添加新功能的步骤
1. 创建新功能分支
2. 实现功能代码
3. 编写必要的文档说明
4. 运行测试确保功能正常
5. 提交代码并创建Pull Request

### 文档维护
每完成一个重要模块或者功能，需要将重要说明添加到 [docs](../docs/) 目录下。