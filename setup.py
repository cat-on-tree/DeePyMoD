# -*- coding: utf-8 -*-
"""
Setup file for DeePyMoD.
"""
import sys
from setuptools import setup

try:
    import setuptools
    from packaging.version import Version
    if Version(setuptools.__version__) < Version("38.3"):
        print("Error: version of setuptools is too old (<38.3)!")
        sys.exit(1)
except Exception:
    # 检查失败时不阻断，交给 pip/setuptools 后续报错
    pass

if __name__ == "__main__":
    setup(use_pyscaffold=True)