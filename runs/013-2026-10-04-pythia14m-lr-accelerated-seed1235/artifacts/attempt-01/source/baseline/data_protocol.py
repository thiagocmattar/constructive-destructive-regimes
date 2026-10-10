"""Verified cache reads and a nonrepeating pass including the short token tail."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_cache(directory, expected_sha256, config):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    path = directory / "tokens.int32.bin"
    if (metadata["dataset_revision"] != config["revision"]
            or metadata["tokenizer_revision"] != config["tokenizer_revision"]
            or metadata["block_size"] != config["sequence_length"]
            or metadata["dtype"] != "int32" or not metadata["append_eos"]):
        raise ValueError("Token-cache protocol differs from the run configuration.")
    if path.stat().st_size != metadata["tokens"] * 4 or sha256(path) != expected_sha256:
        raise ValueError("Token-cache byte count/hash differs from the pinned input.")
    return np.memmap(path, mode="r", dtype="<i4"), metadata


def block_order(token_count, length, seed):
    complete, tail = divmod(token_count, length)
    if tail == 1:
        raise ValueError("A one-token tail has no next-token target; define its handling explicitly.")
    order = np.random.default_rng(seed).permutation(complete).astype(np.int64)
    if tail:
        order = np.concatenate([order, np.array([complete], dtype=np.int64)])
    return order


def update_microbatches(tokens, indices, length, microbatch_size, device):
    """Yield same-length groups, separating a short tail to avoid masked padding."""
    indices = np.asarray(indices, dtype=np.int64)
    lengths = np.minimum(length, len(tokens) - indices * length)
    start = 0
    while start < len(indices):
        stop = min(start + microbatch_size, len(indices))
        for i in range(start + 1, stop):
            if lengths[i] != lengths[start]:
                stop = i
                break
        size = int(lengths[start])
        offsets = indices[start:stop, None] * length + np.arange(size)[None, :]
        batch = torch.from_numpy(np.asarray(tokens[offsets], dtype=np.int64)).to(device)
        yield batch, batch.shape[0] * (size - 1)
        start = stop


def target_count(token_count, indices, length):
    return int(np.sum(np.minimum(length, token_count - np.asarray(indices) * length) - 1))
