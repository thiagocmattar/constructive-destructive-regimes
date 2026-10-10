"""Accept transport only if decompressed bytes match the pinned token cache."""
import gzip
import hashlib
import json
from pathlib import Path
import time

start = time.monotonic()
root = Path('/workspace/minipile-cache/train')
temporary = root / 'tokens.int32.bin.tmp'
digest = hashlib.sha256()
size = 0
with gzip.open(root / 'tokens.gz', 'rb') as incoming, temporary.open('wb') as outgoing:
    while chunk := incoming.read(16*1024*1024):
        outgoing.write(chunk)
        digest.update(chunk)
        size += len(chunk)
expected = json.loads(Path(__file__).with_name('config.json').read_text())['data']['train_sha256']
if digest.hexdigest() != expected or size != 5966845664:
    raise RuntimeError(f'Input hash/size mismatch: {digest.hexdigest()}, {size}')
temporary.replace(root / 'tokens.int32.bin')
receipt = {'accepted':True,'bytes':size,'sha256':digest.hexdigest(),
           'seconds':time.monotonic()-start,'compressed_bytes':(root/'tokens.gz').stat().st_size}
Path('/workspace/constructive/input-transfer-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
