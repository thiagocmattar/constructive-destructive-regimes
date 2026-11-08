"""Run279 entrypoint; frozen inputs pin the shared controller."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks.functional_budget_continuation import main
if __name__=="__main__":main()
