set -uo pipefail
bash /workspace/cdr215-control/setup.sh > /workspace/cdr215-control/setup.log 2>&1
if test "$(cat /workspace/cdr215-control/setup.exit)" = 0; then
 bash /workspace/cdr215-control/worker.sh > /workspace/cdr215-control/worker.log 2>&1
else
 echo 125 > /workspace/cdr215-control/worker.exit
fi
