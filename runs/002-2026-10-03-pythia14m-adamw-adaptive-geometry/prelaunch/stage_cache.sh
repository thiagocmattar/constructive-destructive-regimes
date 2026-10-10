#!/usr/bin/env bash
set -euo pipefail
mkdir -p /opt/minipile-cache/train /opt/minipile-cache/validation
cp /workspace/minipile-cache/train/metadata.json /opt/minipile-cache/train/
cp /workspace/minipile-cache/train/tokens.int32.bin /opt/minipile-cache/train/
cp /workspace/minipile-cache/validation/metadata.json /opt/minipile-cache/validation/metadata.json
cp /workspace/minipile-cache/validation/tokens.int32.bin /opt/minipile-cache/validation/tokens.int32.bin
sha256sum /opt/minipile-cache/train/tokens.int32.bin /opt/minipile-cache/validation/tokens.int32.bin
echo STAGING_COMPLETE
