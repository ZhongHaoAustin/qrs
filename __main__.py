#!/usr/bin/env python3
"""
主模块入口点 - 使测试程序可以作为模块运行
使用方式: python -m test_pattern_recognition_real_data
"""

from pathlib import Path
import sys

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入并运行测试程序
if __name__ == "__main__":
    try:
        from test_pattern_recognition_real_data import main

        print("=" * 60)
        print("启动模式识别研究测试（模块方式运行）")
        print("=" * 60)
        result = main()
        print("=" * 60)
        print("测试完成")
        print("=" * 60)
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖模块都已正确安装")
        sys.exit(1)
    except Exception as e:
        print(f"运行错误: {e}")
        sys.exit(1)
