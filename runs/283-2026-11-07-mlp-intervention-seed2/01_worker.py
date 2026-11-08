"""Pinned-seed MLP replication entrypoint."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks.seeded_budget_continuation import main
if __name__=="__main__":main()
