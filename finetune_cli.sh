#!/usr/bin/env bash
# Launch the warehouse-dispatch fine-tune on Fireworks via firectl.
# Run this the moment Fireworks enables fine-tune/inference provisioning on the
# account (until then it fails with: "unkey inference api id is not configured"
# -- a Fireworks backend issue, fixed only by their support/Discord).
#
#   bash finetune_cli.sh
#
# firectl was downloaded to /tmp/firectl (firectl 1.7.24). If that's gone:
#   curl -fsSL -o /tmp/firectl.gz \
#     https://storage.googleapis.com/fireworks-public/firectl/stable/linux-amd64.gz
#   gunzip -f /tmp/firectl.gz && chmod +x /tmp/firectl
set -euo pipefail
cd "$(dirname "$0")"

FIRECTL="${FIRECTL:-/tmp/firectl}"
source .venv/bin/activate
FW=$(python -c "from env_setup import load_keys; load_keys(); import os; print(os.environ['FIREWORKS_API_KEY'])")

"$FIRECTL" create sftj \
  --base-model accounts/fireworks/models/gemma-4-26b-a4b-it \
  --dataset environment-fine-tuning \
  --epochs 3 \
  --max-context-length 4096 \
  --lora-rank 8 \
  --output-model warehouse-dispatch-ft \
  --eval-auto-carveout \
  --api-key "$FW"

echo
echo "If it succeeded, watch it with:"
echo "  $FIRECTL list sftj --api-key \$FW"
echo "When COMPLETED, the model id is:"
echo "  accounts/brillant16-gcv-48t7k/models/warehouse-dispatch-ft"
echo "Then prove the improvement:"
echo "  python eval_agents.py --fireworks \\"
echo "    --ft-model accounts/brillant16-gcv-48t7k/models/warehouse-dispatch-ft --episodes 3"
