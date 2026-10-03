"""pytest 共享路径配置：让 `import src.xxx` 在任何工作目录下都能解析。"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
