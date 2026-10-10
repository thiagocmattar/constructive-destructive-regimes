import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from baseline.temporal_sketch import TemporalRecorder

def test_exact_window_identity_and_preclip_capture(tmp_path):
    parameter=torch.nn.Parameter(torch.zeros(4))
    config={'temporal_sketch':{'dimension':64,'seeds':[11,12],
        'anchor_seed':13,'anchor_block_updates':16,'exact_lags':[2,4],
        'windows':{'early':[1,3]}}}
    rng_state=torch.get_rng_state().clone()
    recorder=TemporalRecorder([('p',parameter)],config,tmp_path)
    assert torch.equal(rng_state,torch.get_rng_state())
    a=torch.tensor([1.,2.,3.,4.]); norm=float(a.double().norm())
    for i,g in enumerate([a,-a,a]):
        parameter.grad=g.clone()
        recorder.consume({'gradient_norm':norm,'adjacent_cosine':None if i==0 else -1.})
        assert torch.equal(parameter.grad,g)
    result=json.loads((tmp_path/'exact_windows.jsonl').read_text())
    energy=float(a.double().square().sum())
    assert abs(result['energy']-3*energy)<1e-10
    assert abs(result['cumulative_squared_norm']-energy)<1e-10
    assert abs(result['all_pair_interaction']+energy)<1e-10
    assert abs(result['lag_one_interaction']+2*energy)<1e-10
    assert abs(result['beyond_lag_one_interaction']-energy)<1e-10
    state=recorder.state_dict()
    assert state['completed_updates']==3 and state['window_cumulative'] is None
    assert len(state['completed_windows'])==1
