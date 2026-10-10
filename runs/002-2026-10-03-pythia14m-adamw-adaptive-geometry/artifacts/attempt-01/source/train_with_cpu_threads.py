"""Preserve matched initialization, then limit CPU threads for GPU dispatch."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import training

original_build_model = training.build_model


def build_model_with_threads(*args, **kwargs):
    model, metadata = original_build_model(*args, **kwargs)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    metadata["execution_cpu_threads_before"] = before
    metadata["execution_cpu_threads"] = torch.get_num_threads()
    metadata["execution_interop_threads"] = torch.get_num_interop_threads()
    return model, metadata


training.build_model = build_model_with_threads
training.main()
