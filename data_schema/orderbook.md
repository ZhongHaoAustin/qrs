# 订单簿数据说明文档

## 数据读取方式

订单簿数据通过 `options_scalping_strategy/data/data_fetching.py` 中的函数进行读取：

```python
from src.data.data_fetching import fetch_tick_data_range
from vnpy.trader.constant import Exchange

# 获取订单簿数据
df = fetch_tick_data_range(
    instrument_id="SSE.10009482",  # 合约代码
    exchange=Exchange.SSE,         # 交易所
    start_date=20250801,          # 开始日期 (YYYYMMDD格式)
    end_date=20250801             # 结束日期 (YYYYMMDD格式)
)
```

## 数据字段说明

### 基本信息
- **数据格式**: DataFrame
- **时间精度**: 毫秒级

### 字段详细说明

#### 时间相关字段
| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| `datetime` | datetime64[ns] | 时间戳，精确到毫秒 |
| `localtime` | object | 本地时间 |

#### 价格相关字段
| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| `last_price` | float64 | 最新成交价 |
| `highest` | float64 | 当日最高价 |
| `lowest` | float64 | 当日最低价 |
| `average` | float64 | 平均价格 |
| `vwap` | float64 | 成交量加权平均价格 |

#### 成交量相关字段
| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| `volume` | float64 | 成交量 |
| `amount` | float64 | 成交金额 |
| `open_interest` | float64 | 持仓量 |

#### 订单簿字段 (5档深度)
| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| `bid_price1` | float64 | 第1档买价 |
| `bid_volume1` | float64 | 第1档买量 |
| `ask_price1` | float64 | 第1档卖价 |
| `ask_volume1` | float64 | 第1档卖量 |
| `bid_price2` | float64 | 第2档买价 |
| `bid_volume2` | float64 | 第2档买量 |
| `ask_price2` | float64 | 第2档卖价 |
| `ask_volume2` | float64 | 第2档卖量 |
| `bid_price3` | float64 | 第3档买价 |
| `bid_volume3` | float64 | 第3档买量 |
| `ask_price3` | float64 | 第3档卖价 |
| `ask_volume3` | float64 | 第3档卖量 |
| `bid_price4` | float64 | 第4档买价 |
| `bid_volume4` | float64 | 第4档买量 |
| `ask_price4` | float64 | 第4档卖价 |
| `ask_volume4` | float64 | 第4档卖量 |
| `bid_price5` | float64 | 第5档买价 |
| `bid_volume5` | float64 | 第5档买量 |
| `ask_price5` | float64 | 第5档卖价 |
| `ask_volume5` | float64 | 第5档卖量 |

#### 其他字段
| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| `instrument_id` | object | 合约代码 |

### 订单簿特点
- 提供5档深度数据
- 买价从高到低排列 (bid_price1 > bid_price2 > ...)
- 卖价从低到高排列 (ask_price1 < ask_price2 < ...)
- 部分档位可能为空值 (NaN)

## 使用示例

```python
import pandas as pd
from data_fetching import fetch_tick_data_range
from vnpy.trader.constant import Exchange

# 获取数据
df = fetch_tick_data_range(
    instrument_id="SSE.10009482",
    exchange=Exchange.SSE,
    start_date=20250801,
    end_date=20250801
)

## 注意事项

1. **数据完整性**: 部分档位的价格和成交量可能为空值 (NaN)
2. **时间精度**: 时间戳精确到毫秒，适合高频交易分析
3. **数据范围**: 仅包含交易时间内的数据
4. **价格单位**: 价格以小数形式表示，如0.0001表示0.0001元
