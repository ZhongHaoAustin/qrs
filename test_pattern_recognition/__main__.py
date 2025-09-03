#!/usr/bin/env python3
"""
模式识别研究测试包的主模块入口点
使用方式: python -m test_pattern_recognition
"""

from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """主测试函数"""
    try:
        # 导入测试模块
        from test_pattern_recognition_real_data import main as run_tests

        print("=" * 60)
        print("启动模式识别研究测试（模块方式运行）")
        print("=" * 60)

        # 运行测试
        result = run_tests()

        print("=" * 60)
        print("测试完成")
        print("=" * 60)

        return result

    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖模块都已正确安装")
        print(f"当前Python路径: {sys.path[:3]}")
        sys.exit(1)
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
