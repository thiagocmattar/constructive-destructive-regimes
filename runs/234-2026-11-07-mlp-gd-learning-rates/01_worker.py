"""Run234: execute independently reset GD rates in the declared order."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'src'))
from eos_tasks.gd_intervention import main, gd_optimizer

if __name__=='__main__':
    main(HERE)
