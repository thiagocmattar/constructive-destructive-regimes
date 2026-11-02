from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks.adam_profile import main
if __name__ == "__main__":main(Path(__file__).parent/"config.json")
