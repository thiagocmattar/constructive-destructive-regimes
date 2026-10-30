"""H024 checks: native-update equivalence, moment timing, displacement and recovery."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from baseline.lr_decisions import (initialize_streams,step,set_rate,save_state,restore_state,cosine,norm)
from baseline.recipe import build_recipe_adamw,DynamicLossScaler
from baseline.gradient_geometry import GradientGeometry
from baseline.training import optimizer_boundary
from baseline.continuation import rng_state,restore_rng,protected_evaluate
from baseline.descent_diagnostics import fixed_probe

def setup():
    torch.set_num_threads(1)
    torch.manual_seed(24)
    model=GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=32,rotary_pct=.5,hidden_dropout=.1,attention_dropout=.1))
    config=dict(data=dict(sequence_length=8),training=dict(optimizer='adamw',peak_learning_rate=.01,minimum_learning_rate=.001,warmup_fraction=.01,adamw_betas=[.9,.95],adamw_eps=1e-8,weight_decay=.1,gradient_clip_norm=.1,micro_batch_size=2,maximum_overflow_retries=2),validation=dict(batch_size=2),decision=dict(base_lr=.01,multiplier=2.))
    optimizer,_=build_recipe_adamw(model,config['training'],torch=torch)
    raw=GradientGeometry(model.named_parameters());adaptive=GradientGeometry(model.named_parameters(),previous_dtype=torch.float64)
    scaler=DynamicLossScaler(scale=1.)
    tokens=np.arange(256,dtype=np.int32)%32
    optimizer_boundary(model,optimizer,scaler,raw,adaptive,tokens,np.arange(4),config,torch.device('cpu'))
    return model,optimizer,scaler,raw,adaptive,tokens,config

def test_instrumentation_matches_native_and_direct_vectors(tmp_path):
    model,opt,scale,raw,adaptive,tokens,c=setup()
    streams=initialize_streams(model,opt,raw,adaptive,1)
    initial=[p.detach().clone() for p in model.parameters()]
    old_m=[opt.state[p]['exp_avg'].clone() for p in model.parameters()]
    native=copy.deepcopy(model)
    native_opt,_=build_recipe_adamw(native,c['training'],torch=torch)
    native_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    ng=GradientGeometry(native.named_parameters());nd=GradientGeometry(native.named_parameters(),previous_dtype=torch.float64)
    ng.load_state_dict(raw.state_dict());nd.load_state_dict(adaptive.state_dict())
    ns=copy.deepcopy(scale)
    set_rate(opt,c,2);set_rate(native_opt,c,2)
    state=rng_state()
    actual=step(model,opt,scale,streams,tokens,np.arange(4,8),c,torch.device('cpu'))
    post_rng=rng_state()
    restore_rng(state)
    expected=optimizer_boundary(native,native_opt,ns,ng,nd,tokens,np.arange(4,8),c,torch.device('cpu'))
    assert actual['loss']==expected['loss']
    for p,q in zip(model.parameters(),native.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),post_rng['torch_rng'])
    assert actual['incoming_previous_moment_cosine']==pytest.approx(cosine(raw.previous,old_m),abs=1e-14)
    delta=[p.double()-b.double() for p,b in zip(model.parameters(),initial)]
    assert actual['actual_parameter_update']['gradient_norm']==pytest.approx(norm(delta),abs=1e-14)
    assert actual['actual_parameter_update']['cumulative_ratio']==pytest.approx(1.)
    assert actual['clipped']['gradient_norm']==pytest.approx(actual['gradient_norm']*actual['clip_coefficient'],rel=2e-7)
    for group in opt.param_groups:
        assert group['lr']*group['weight_decay'] in (0.,.001)
    path=tmp_path/'state.pt'
    save_state(path,model,opt,scale,streams,8,2,'order','config')
    # Full validation and fixed probes must preserve training randomness and gradients.
    saved_grads=[p.grad.clone() for p in model.parameters()]
    protected_evaluate(model,tokens[-19:],c,torch.device('cpu'))
    fixed_probe(model,tokens,np.array([20,21]),8,2,torch.device('cpu'))
    assert torch.equal(torch.get_rng_state(),post_rng['torch_rng'])
    for p,g in zip(model.parameters(),saved_grads):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    first=step(model,opt,scale,streams,tokens,np.arange(8,12),c,torch.device('cpu'))
    params=[p.detach().clone() for p in model.parameters()]
    pos,update=restore_state(path,model,opt,scale,streams,'order','config')
    assert (pos,update)==(8,2)
    repeat=step(model,opt,scale,streams,tokens,np.arange(8,12),c,torch.device('cpu'))
    for key in ['loss','gradient_norm','adjacent_cosine','incoming_previous_moment_cosine','projected_gradient_change']:
        assert first[key]==repeat[key]
    assert first['actual_parameter_update']==repeat['actual_parameter_update']
    for p,v in zip(model.parameters(),params):torch.testing.assert_close(p,v,rtol=0,atol=0)

def test_zero_and_negative_alignment():
    zero=[torch.zeros(4)];positive=[torch.ones(4)];negative=[-torch.ones(4)]
    assert cosine(zero,positive) is None
    assert cosine(positive,negative)==-1
    assert cosine(positive,positive)==1


def test_complete_checkpoint_loader_and_artifact_smoke(tmp_path):
    import argparse
    import hashlib
    import json
    from baseline.lr_decisions import execute
    from baseline.data_protocol import block_order,sha256
    from baseline.training import checkpoint
    model,opt,scaler,raw,adaptive,tokens,c=setup()
    order=block_order(len(tokens),8,24)
    order_hash=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
    c['data'].update(revision='synthetic-revision',tokenizer_revision='synthetic-tokenizer')
    for split,array in [('train',tokens),('validation',tokens[:19])]:
        directory=tmp_path/'cache'/split;directory.mkdir(parents=True)
        array.astype('<i4').tofile(directory/'tokens.int32.bin')
        metadata=dict(dataset_revision='synthetic-revision',tokenizer_revision='synthetic-tokenizer',block_size=8,dtype='int32',append_eos=True,tokens=len(array))
        (directory/'metadata.json').write_text(json.dumps(metadata))
        c['data'][split+'_sha256']=sha256(directory/'tokens.int32.bin')
    checkpoint(tmp_path/'parent','midpoint',model,opt,scaler,raw,adaptive,4,order_hash,'synthetic-parent')
    parent=tmp_path/'parent/checkpoints/midpoint'
    c.update(run_id='synthetic-smoke',model=dict(parameter_count=sum(p.numel() for p in model.parameters())),seeds=dict(model=24,data_order=24))
    c['training']['effective_batch_sequences']=4
    c['resume']=dict(junction_update=1,full_updates=8,parent_config_sha256='synthetic-parent',order_sha256=order_hash,checkpoint_sha256={p.relative_to(parent).as_posix():sha256(p) for p in parent.rglob('*') if p.is_file()})
    probe=tmp_path/'probe.json';probe.write_text(json.dumps(dict(indices=order[20:22].tolist(),supervised_tokens=14)))
    c['decision'].update(updates=2,probe_path=str(probe),probe_sha256=sha256(probe),validation_local_updates=[0,2],probe_local_updates=[0,2],checkpoint_local_updates=[])
    config=tmp_path/'config.json';config.write_text(json.dumps(c))
    output=tmp_path/'output'
    execute(argparse.Namespace(config=str(config),output=str(output),cache=str(tmp_path/'cache'),device='cpu',parent=str(parent),resume=None,mechanics=2,replay=False,deadline=None))
    summary=json.loads((output/'summary.json').read_text())
    assert summary['actual_displacement_relative_residual']<1e-10
    assert summary['validations'][0]['supervised_tokens']==16
    assert summary['last_update']['continuation_input_tokens']==64
    assert json.loads((output/'manifest.json').read_text())['status']=='completed'
