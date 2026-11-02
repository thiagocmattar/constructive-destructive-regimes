set -euo pipefail
trap 'echo $? > /workspace/cdr215-control/setup.exit' EXIT
echo '02ebfee8452b70d085df81cbdb8c5c7cd6eaab3a79cecb965b3c0786beeaa859  /workspace/cdr215-inputs.tar.gz' | sha256sum -c -
mkdir -p /workspace/constructive
tar -xzf /workspace/cdr215-inputs.tar.gz -C /workspace/constructive
python3 -m venv --system-site-packages --without-pip /opt/cdr215-venv
python3 -m pip --python /opt/cdr215-venv install --no-cache-dir --disable-pip-version-check numpy==2.5.0 scipy==1.18.1
/opt/cdr215-venv/bin/python -c "import torch,numpy,scipy; assert torch.__version__.split('+')[0]=='2.11.0'; assert torch.cuda.is_available(); print(torch.__version__,numpy.__version__,scipy.__version__,torch.cuda.get_device_name())"
