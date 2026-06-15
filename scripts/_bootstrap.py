"""Make ``src/`` importable when running scripts without ``pip install -e .``."""
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
