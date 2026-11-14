import os
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("pysndlib")
except PackageNotFoundError:  # e.g. running straight from an unbuilt source tree
    __version__ = "0.0.0.dev0"

def get_include():
    import pysndlib
    import os.path
    return os.path.dirname(pysndlib.__file__)
