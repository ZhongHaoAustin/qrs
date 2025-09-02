# orderbook数据研究框架

# data
orderbook数据相关见[data](../data_schema/orderbook.md)

策略从模式识别的角度思考，比如基于orderbook某类别情况识别出来，基于该情况之前的数据构建特征，基于该情况之后的数据构建评价指标，一步步去识别模式和细化特征，提升评估指标，基于这个需求，需要实现三个抽象的模块并且给出一些case，1. 模式识别模块，2. 特征构建模块，3. 评价指标模块

要求：
- 模块化编程，方便复用，使用FP编程风格，main函数接受[配置文件](../config/xxx.yaml)，然后把所有函数compose起来，最后运行。
- 每个模块可以通过shell命令行接受指定参数运行测试

## 文档规范
每完成一个重要模块或者功能，把重要说明吸到[文档目录下](../docs/)

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


## 配置管理
配置文件应使用YAML格式，包含必要的注释，[配置文件存放目录](../config/)。


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