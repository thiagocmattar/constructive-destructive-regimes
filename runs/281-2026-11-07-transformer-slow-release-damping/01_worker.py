"""Compose two verified probe policies; authenticate and reuse the native control."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks import representation_continuation as c
from src.eos_tasks import functional_budget_continuation as budget
from src.eos_tasks import functional_acceptance_continuation as guard
from src.eos_tasks import reuse_control

def main():
    if "--selftest" in sys.argv:
        budget.main();guard.main();return
    reuse_control.install()
    native_main=c.main
    def enter_guard():
        c.main=native_main
        guard.main()
    c.main=enter_guard
    budget.main()

if __name__=="__main__":main()
