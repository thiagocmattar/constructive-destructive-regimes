"""MLP replication from pinned existing initializations; preserve all other inputs."""
import argparse
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from src.eos_tasks import functional_budget_continuation as budget


def main():
    if '--selftest' in sys.argv:budget.main();return
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--run',type=Path,required=True);args,_=parser.parse_known_args()
    cfg=json.loads((args.run/'config.json').read_text());original=budget.c.w.prepare
    assert cfg['case']=='mlp' and cfg['seed'] in [1,2]
    def prepare(case,parent):
        context=original(case,parent);seed=cfg['seed']
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        state=torch.load(budget.ROOT/cfg['initial_state'],map_location='cpu',weights_only=True)
        context['task'].model.load_state_dict(state)
        assert budget.c.w.tensor_hash(context['task'].model.parameters())==cfg['initial_tensor_sha256']
        assert not context['opt'].state and context['state']==0
        return context
    budget.c.w.prepare=prepare;budget.main()


if __name__=='__main__':main()
