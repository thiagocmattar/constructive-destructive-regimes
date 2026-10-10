"""Keep compressed input durable; decompress identical token bytes on local disk."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

start = time.monotonic()
run = Path(__file__).resolve().parents[1]
config = json.loads((run / "config.json").read_text())
durable, local = Path("/workspace/minipile-cache"), Path("/opt/minipile-cache")
for split in ("train", "validation"):
    target = local / split
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(durable / split / "metadata.json", target / "metadata.json")
    if split == "train":
        digest, size = hashlib.sha256(), 0
        with gzip.open(durable / split / "tokens.gz", "rb") as incoming, (target / "tokens.int32.bin").open("wb") as outgoing:
            while chunk := incoming.read(16 * 1024 * 1024):
                outgoing.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != config["data"]["train_sha256"] or size != 5966845664:
            raise RuntimeError("Decompressed training bytes differ from pinned source.")
    else:
        shutil.copy2(durable / split / "tokens.int32.bin", target / "tokens.int32.bin")
        with (target / "tokens.int32.bin").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != config["data"]["validation_sha256"]:
                raise RuntimeError("Validation bytes differ from pinned source.")
receipt = {"accepted": True, "local_training_cache": str(local),
           "durable_train_master": str(durable / "train/tokens.gz"),
           "bytes": size, "sha256": digest.hexdigest(), "seconds": time.monotonic() - start,
           "compressed_bytes": (durable / "train/tokens.gz").stat().st_size,
           "validation_sha256": config["data"]["validation_sha256"]}
Path("/workspace/constructive/input-transfer-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
