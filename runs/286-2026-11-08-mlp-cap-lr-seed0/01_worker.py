"""H049 MLP distributed-cap factorial entrypoint."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks.distributed_cap_sweep import main
if __name__=="__main__": main()
