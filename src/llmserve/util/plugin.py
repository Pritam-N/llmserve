# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import importlib

class PluginLoadError(Exception):
    pass

def load_symbol(module_path: str, symbol: str):
    try:
        mod = importlib.import_module(module_path)
        if not hasattr(mod, symbol):
            raise PluginLoadError(f"{module_path} has no symbol {symbol}")
        return getattr(mod, symbol)
    except Exception as e:
        raise PluginLoadError(f"failed to load {symbol} from {module_path}: {e}") from e