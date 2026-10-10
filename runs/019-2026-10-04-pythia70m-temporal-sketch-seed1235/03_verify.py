import sys,json
from pathlib import Path
import numpy as np
from verify_baseline import verify
root=Path(sys.argv[1]); base=verify(root)
config=json.loads((root/'resolved_config.json').read_text())
history=np.load(root/'temporal/raw_sketches.npy',mmap_mode='r')
assert history.shape==(712,4,16384) and history.dtype==np.float32
assert np.isfinite(history).all()
assert json.loads((root/'temporal/sketch_progress.json').read_text())['completed_updates']==712
cal=json.loads((root/'temporal/calibration.json').read_text())
assert cal['qualified'] and cal['completed_updates']==712
assert cal['exact_pairs']>900
print(json.dumps({**base,'temporal_verified':True,'temporal_calibration':cal},indent=2))
