from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from src.eos_tasks.selective_continuation import main
if __name__=="__main__":main(HERE)
