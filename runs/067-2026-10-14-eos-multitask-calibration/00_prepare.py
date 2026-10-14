"""Prepare complete paper objectives and a pinned pre-submission LM reference."""
import io
import json
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
import requests
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, tensor_hash, write_json
from eos import model_for

CACHE = ROOT / "cache/eos-multitask"
TUTORIAL = "13acec6d7c78dacd5e1fe9b0b4a325e1d39abc15"


def language_chunks(tokens, streams):
    omitted = len(tokens) % streams
    matrix = tokens[:len(tokens)-omitted if omitted else len(tokens)].reshape(streams, -1).T
    xs, ys = [], []
    for start in range(0, len(matrix)-1, 35):
        size = min(35, len(matrix)-1-start)
        x = torch.zeros((streams, 35), dtype=torch.int64)
        y = torch.full_like(x, -100)
        x[:, :size] = matrix[start:start+size].T
        y[:, :size] = matrix[start+1:start+1+size].T
        xs.append(x); ys.append(y)
    return torch.cat(xs), torch.cat(ys), {"original_tokens": len(tokens), "batchify_tail_omitted": omitted,
        "stream_initial_tokens_without_target": streams, "target_tokens": sum(int((a != -100).sum()) for a in ys),
        "sequence_units": sum(len(a) for a in xs), "length": 35, "batchify_streams": streams}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "inputs.json").exists():
        raise FileExistsError("Prepared scientific inputs already frozen.")
    archive = ROOT / "cache/cifar10-eos/cifar-10-python.tar.gz"
    assert file_hash(archive, "md5") == "c58f30108f718f92721af3b95e74349a"
    with tarfile.open(archive) as bundle:
        def batch(name):
            return pickle.load(io.BytesIO(bundle.extractfile("cifar-10-batches-py/"+name).read()), encoding="bytes")
        parts = [batch(f"data_batch_{i}") for i in range(1,6)]
        raw = np.concatenate([p[b"data"] for p in parts]).reshape(-1,3,32,32).astype(np.float64)/255
        labels = np.concatenate([p[b"labels"] for p in parts])
        test = batch("test_batch")
    mean = raw.mean(axis=(0,2,3), keepdims=True)
    std = (raw-mean).std(axis=(0,2,3), keepdims=True)
    prepared = {"x_train": torch.from_numpy(((raw-mean)/std).astype(np.float32)), "y_train": torch.tensor(labels),
        "x_test": torch.from_numpy(((test[b"data"].reshape(-1,3,32,32)/255-mean)/std).astype(np.float32)),
        "y_test": torch.tensor(test[b"labels"])}
    torch.save(prepared, CACHE / "resnet_data.pt")
    del raw, prepared
    wiki = CACHE / "wikitext"
    wiki.mkdir(exist_ok=True)
    vocabulary = {}
    language, coverage, sources = {}, {}, {}
    for split, filename, streams in [("train", "train.txt",20),("valid","valid.txt",10),("test","test.txt",10)]:
        path = wiki / filename
        url = f"https://raw.githubusercontent.com/pytorch/examples/{TUTORIAL}/word_language_model/data/wikitext-2/{filename}"
        if not path.exists():
            result = requests.get(url, timeout=90)
            result.raise_for_status()
            path.write_bytes(result.content)
        tokens = []
        for line in path.read_text(encoding="utf-8").splitlines():
            for word in line.split()+["<eos>"]:
                if word not in vocabulary:
                    vocabulary[word] = len(vocabulary)
                tokens.append(vocabulary[word])
        x, y, info = language_chunks(torch.tensor(tokens), streams)
        language["x_"+split], language["y_"+split] = x, y
        coverage[split] = info
        sources[split] = {"url": url, "sha256": file_hash(path), "bytes": path.stat().st_size}
    torch.save(language, CACHE / "transformer_data.pt")
    (CACHE / "vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
    initial = {}
    for task in ("mlp","resnet","transformer"):
        initial[task] = []
        for seed in (0,1,2):
            torch.manual_seed(seed)
            model = model_for(task, CACHE, len(vocabulary))
            path = CACHE / f"{task}_initial_seed{seed}.pt"
            torch.save(model.state_dict(), path)
            initial[task].append({"seed": seed, "path": path.name, "sha256": file_hash(path),
                "initial_tensor_sha256": tensor_hash(model.parameters()), "parameters": sum(p.numel() for p in model.parameters())})
    # The seed0 MLP initialization must exactly match the previous frozen baseline.
    previous = json.loads((ROOT / "cache/cifar10-eos/inputs.json").read_text())
    assert initial["mlp"][0]["initial_tensor_sha256"] == previous["initial_tensor_sha256"]
    refs = json.loads((CACHE / "reference/sources.json").read_text())
    for name in refs:
        refs[name]["sha256"] = file_hash(CACHE / "reference" / name)
    record = {"reference_sources": refs, "cifar_archive_sha256": file_hash(archive), "cifar_normalization":
        {"resnet": "full50k channel population mean/std", "mlp": "unchanged Run058 released-code coordinate normalization"},
        "wikitext": {"vocabulary_size": len(vocabulary), "dictionary": "tutorial train/valid/test first-occurrence union",
                     "sources": sources, "coverage": coverage}, "initial": initial,
        "prepared_files": {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in CACHE.glob("*.pt")}}
    write_json(CACHE / "inputs.json", record)
    write_json(HERE / "inputs.json", record)
    print(json.dumps({"vocabulary": len(vocabulary), "coverage": coverage, "initial": initial}, indent=2))


if __name__ == "__main__":
    main()
