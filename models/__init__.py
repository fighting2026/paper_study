# -*- coding: utf-8 -*-
"""
models 包

仅包含 Mamba 选择性扫描（selective scan）的底层算子实现：
- models/csms6s.py  —— SelectiveScanMamba / SelectiveScanCore / SelectiveScanOflex

注意：本文件原本会 `from .vmamba import VSSM`，但本项目并未使用 VMamba，
为了让 `from models.csms6s import ...` 可以正常导入，这里保持为空。
"""
