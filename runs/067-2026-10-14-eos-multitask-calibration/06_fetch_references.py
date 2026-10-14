"""Recreate pinned source prerequisites without changing an existing cache."""
import json
import sys
from pathlib import Path

import requests

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash


def main():
    sources=json.loads((HERE/"inputs.json").read_text())["reference_sources"]
    folder=ROOT/"cache/eos-multitask/reference"
    folder.mkdir(parents=True,exist_ok=True)
    for name,source in sources.items():
        path=folder/name
        if not path.exists():
            url=f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{source['path']}"
            result=requests.get(url,timeout=60);result.raise_for_status();path.write_bytes(result.content)
        assert file_hash(path)==source["sha256"],name
    metadata=folder/"sources.json"
    if not metadata.exists():
        metadata.write_text(json.dumps({name:{key:value for key,value in s.items() if key!="sha256"}
            for name,s in sources.items()},indent=2),encoding="utf-8")
    print("Pinned historical reference prerequisites verified.")


if __name__=="__main__":main()
