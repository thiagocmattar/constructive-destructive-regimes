"""Run216 uses Run214's frozen implementation with its own immutable config."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'runs/214-2026-11-02-sgd-batch-extension'/'01_measure.py'
spec=importlib.util.spec_from_file_location('run216_impl_01_measure.py',SOURCE)
implementation=importlib.util.module_from_spec(spec);spec.loader.exec_module(implementation)
implementation.HERE=HERE
implementation.ROOT=ROOT

if __name__=='__main__':implementation.main()
