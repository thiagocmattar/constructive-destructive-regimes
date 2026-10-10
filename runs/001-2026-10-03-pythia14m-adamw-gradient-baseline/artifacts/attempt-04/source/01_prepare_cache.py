"""Rebuild the pinned MiniPile cache on the worker and require the reference hashes."""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer


def prepare(root, config):
    root = Path(root)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["architecture"],
                         revision=config["data"]["tokenizer_revision"])
    tokenizer.model_max_length = 10**12
    for split in ("validation", "train"):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "tokens.int32.bin"
        metadata_path = directory / "metadata.json"
        expected = config["data"][f"{split}_sha256"]
        if metadata_path.exists():
            print(f"Existing {split} cache; training will independently verify bytes.", flush=True)
            continue
        rows = load_dataset(config["data"]["dataset"], revision=config["data"]["revision"],
                            split=split, streaming=True)
        documents = tokens = 0
        digest = hashlib.sha256()
        buffer = []
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            for row in rows:
                if not row["text"]:
                    continue
                ids = tokenizer.encode(row["text"], add_special_tokens=False)
                ids.append(tokenizer.eos_token_id)
                buffer.extend(ids)
                documents += 1
                tokens += len(ids)
                if len(buffer) >= 1_000_000:
                    blob = np.asarray(buffer, dtype="<i4").tobytes()
                    stream.write(blob); digest.update(blob); buffer.clear()
                if documents % 10000 == 0:
                    print(json.dumps({"split": split, "documents": documents, "tokens": tokens}), flush=True)
            blob = np.asarray(buffer, dtype="<i4").tobytes()
            stream.write(blob); digest.update(blob)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(f"Rebuilt {split} cache hash differs: {actual}")
        temporary.replace(path)
        complete, tail = divmod(tokens, config["data"]["sequence_length"])
        metadata = {"dataset_name": config["data"]["dataset"],
                    "dataset_revision": config["data"]["revision"], "split": split,
                    "tokenizer_name": config["model"]["architecture"],
                    "tokenizer_revision": config["data"]["tokenizer_revision"],
                    "documents": documents, "tokens": tokens, "tokens_bytes": tokens * 4,
                    "tokens_sha256": actual, "tokens_path": "tokens.int32.bin", "dtype": "int32",
                    "append_eos": True, "block_size": config["data"]["sequence_length"],
                    "complete_blocks": complete, "excluded_tail_tokens": tail}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = parser.parse_args()
    prepare(args.output, json.loads(Path(args.config).read_text(encoding="utf-8")))
