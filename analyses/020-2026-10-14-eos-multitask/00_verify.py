"""Check new vector-variance reductions and official architecture parity."""
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.eos_tasks.task import Task,configure,model_for,module
from src.eos_tasks.noise import mlp_unit_energy,variance_probe
from src.eos_tasks.production import ProductionTask
from src.cifar_eos.core import flat,make_model,write_json,full_gradient,file_hash


def main():
    configure()
    torch.manual_seed(7)
    model=make_model(input_size=3,widths=(7,5)).double()
    data={"x_train":torch.randn(13,3,dtype=torch.float64),"y_train":torch.arange(13)%10}
    task=Task("mlp",model,data,4)
    _,reference=task.gradient()
    production=ProductionTask("mlp",model,data,4)
    delegated_loss,delegated=production.gradient()
    original_loss,_,original=full_gradient(model,data["x_train"],data["y_train"],4,"mse")
    assert delegated_loss==original_loss and torch.equal(delegated,original)
    pieces=torch.stack([task.gradient(torch.tensor([i]))[1] for i in range(13)])
    energy=float(pieces.square().sum())
    assert abs(energy-mlp_unit_energy(task))<1e-10
    generator=torch.Generator().manual_seed(11)
    probe=variance_probe(task,generator,reference,3)
    variance=float((pieces-reference).square().sum()/12)
    assert abs(probe["trace_population_sample_covariance"]-variance)<1e-10
    cache=ROOT/"cache/eos-multitask"
    torch.manual_seed(7)
    adapted=model_for("resnet",cache)
    original=module(cache/"reference/resnet_cifar.py","original_resnet").resnet32()
    original.load_state_dict({**adapted.state_dict(),**{k:v for k,v in original.state_dict().items()
        if k.endswith(("running_mean","running_var","num_batches_tracked"))}})
    adapted.train();original.train()
    with torch.no_grad():
        x=torch.randn(4,3,32,32)
        assert torch.equal(adapted(x),original(x))
    del adapted,original
    manifest=json.loads((cache/"inputs.json").read_text())
    lm=model_for("transformer",cache,manifest["wikitext"]["vocabulary_size"])
    lm.load_state_dict(torch.load(cache/"transformer_initial_seed0.pt",weights_only=True))
    language={"x_train":torch.randint(0,100,(3,35)),"y_train":torch.randint(0,100,(3,35))}
    language["y_train"][-1,10:]=-100
    t=Task("transformer",lm,language,2)
    value,output=t.loss(language["x_train"],language["y_train"])
    independently=torch.nn.functional.nll_loss(lm(language["x_train"].T).reshape(-1,manifest["wikitext"]["vocabulary_size"]),
        language["y_train"].T.reshape(-1),ignore_index=-100,reduction="sum")
    assert torch.allclose(value,independently,atol=1e-3,rtol=1e-6)
    assert t.units==80
    write_json(Path(__file__).parent/"mathematical_verification.json",{"status":"passed",
        "exact_mlp_per_example_gradient_energy":True,"finite_population_batch_variance":True,
        "stateless_ghost_bn_forward_parity":True,"historical_lm_axis_token_weighting":True,
        "padded_targets_excluded":True,"calibration":"Run067 independent HVP/noise/accumulation checks passed"})
    record=json.loads((Path(__file__).parent/"mathematical_verification.json").read_text())
    record.update(existing_mlp_full_gradient_bitwise_parity=True,
        production_source_sha256={p.name:file_hash(p) for p in (ROOT/"src/eos_tasks").glob("*.py")})
    write_json(Path(__file__).parent/"mathematical_verification.json",record)
    print("Production variance and objective-parity checks passed.")


if __name__=="__main__":main()
