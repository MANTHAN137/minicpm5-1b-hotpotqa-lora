"""Continue the active local training job with evaluation and documentation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    status_path = root / "workflow_state.json"

    def state(phase, **extra):
        status_path.write_text(json.dumps({"phase": phase, **extra}, indent=2) + "\n")
        print(phase, flush=True)

    state("waiting_for_training", pid=args.wait_pid)
    while True:
        try:
            os.kill(args.wait_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(15)
    # Require the last checkpoint, not merely an intermediate adapter file.
    final_checkpoint = root / "adapters" / "0002500_adapters.safetensors"
    if not final_checkpoint.exists():
        state("training_incomplete", error="Final 2,500-step checkpoint is missing; evaluation was not started")
        raise SystemExit(1)
    script = Path(__file__).with_name("hotpotqa_lora.py")
    try:
        for phase in ("after", "report"):
            state(phase)
            subprocess.run([sys.executable, str(script), phase, "--output", str(root)], check=True)
    except Exception as exc:
        state("failed", error=str(exc))
        raise
    state("complete", report=str(root / "experiment_report.md"))


if __name__ == "__main__":
    main()
