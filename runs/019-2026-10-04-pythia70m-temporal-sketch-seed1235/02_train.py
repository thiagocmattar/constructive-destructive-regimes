from pathlib import Path
import sys,json
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from baseline import training
from baseline.gradient_geometry import GradientGeometry
from baseline.temporal_sketch import TemporalRecorder,calibration_report
config=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
output=Path(sys.argv[sys.argv.index('--output')+1])
recorders=[]
class RecordingGeometry(GradientGeometry):
    def __init__(self,named_parameters,previous_dtype=torch.float32):
        super().__init__(named_parameters,previous_dtype=previous_dtype)
        self.recorder=None
        if previous_dtype==torch.float32:
            self.recorder=TemporalRecorder(self.parameters,config,output/'temporal')
            recorders.append(self.recorder)
    def consume(self,vectors=None):
        result=super().consume(vectors)
        if self.recorder is not None:
            assert vectors is None
            result.update(self.recorder.consume(result))
        return result
    def state_dict(self):
        result=super().state_dict()
        if self.recorder is not None: result['temporal']=self.recorder.state_dict()
        return result
training.GradientGeometry=RecordingGeometry
if __name__=='__main__':
    training.main()
    assert len(recorders)==1
    report=calibration_report(output/'temporal',recorders[0].count)
    (output/'temporal/calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'temporal_calibration':report}),flush=True)
