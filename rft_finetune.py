"""
rft_finetune.py
===============
RFT step 2/3 -- validate the dataset and launch a Fireworks fine-tune.

Fireworks supervised fine-tuning is driven by their `firectl` CLI (most reliable)
or the dashboard. This script validates the JSONL is in the right chat format,
then prints the exact commands / steps. After the job finishes, set the new
model id in .env as FIREWORKS_MODEL and re-run eval_agents.py (step 3/3).

Usage:
    python rft_finetune.py
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "warehouse_dispatch_sft.jsonl"


def validate(path):
    if not path.exists():
        print(f"Dataset not found: {path}\nRun: python rft_dataset.py")
        return 0
    n, bad = 0, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                msgs = obj["messages"]
                roles = [m["role"] for m in msgs]
                assert roles == ["system", "user", "assistant"]
                json.loads(msgs[2]["content"])  # assistant must be valid JSON
                n += 1
            except Exception:
                bad += 1
    print(f"Validated {path.name}: {n} good examples, {bad} malformed.")
    return n


def main():
    n = validate(DATA)
    if n == 0:
        return
    has_firectl = shutil.which("firectl") is not None

    print("\n" + "=" * 66)
    print("FINE-TUNE ON FIREWORKS  (uses your $506 credits)")
    print("=" * 66)

    print("\nOption A - dashboard (most reliable):")
    print("  1. app.fireworks.ai -> Datasets -> Upload -> choose")
    print(f"     {DATA}")
    print("  2. Fine Tuning -> Supervised Fine Tuning -> new job:")
    print("       base model : pick one tagged 'Tunable (LoRA)' in the Model")
    print("                    Library (e.g. Kimi K2.7 Code, Nemotron 3 Ultra BF16)")
    print("       dataset    : the one you just uploaded")
    print("  3. Start. When done, copy the fine-tuned model id.")

    print("\nOption B - firectl CLI:")
    if not has_firectl:
        print("  (firectl not installed) install it first:")
        print("    curl -fsSL https://storage.googleapis.com/fireworks-public/firectl/stable/firectl.gz | gunzip > firectl && chmod +x firectl")
        print("    ./firectl signin")
    print("    firectl create dataset warehouse-dispatch " + str(DATA))
    print("    firectl create sftj \\")
    print("        --base-model accounts/fireworks/models/kimi-k2p7-code \\")
    print("        --dataset warehouse-dispatch \\")
    print("        --output-model warehouse-dispatch-ft")
    print("    # flags vary by version: firectl create sftj --help")

    print("\nSTEP 3/3 - prove it improved:")
    print("  Put the fine-tuned id in .env:")
    print("    FIREWORKS_MODEL=accounts/<your-account>/models/warehouse-dispatch-ft")
    print("  Then:")
    print("    python eval_agents.py --fireworks --episodes 3")
    print("  Compare its orders/missing vs the base model + heuristic.")
    print("=" * 66)


if __name__ == "__main__":
    main()
