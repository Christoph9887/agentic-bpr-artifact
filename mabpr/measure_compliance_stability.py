"""How often does the compliance gate let AE-001 through? Repeated, gate-only."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mabpr import datasets
from mabpr.agents.arbiter import ArbiterBox, gate

TARGET = "AE-001"

def main(dataset: str | None = None, repeats: int = 11) -> None:
    ds = datasets.get(dataset)
    cands = json.loads(ds.detector_results.read_text(encoding="utf-8"))
    ordering = [c["candidate_id"] for c in cands]

    print(f"dataset {ds.tag}: {len(ordering)} candidates, asking the gate "
          f"{repeats} times about the SAME plan")
    print(f"  plan: {ordering}\n")

    outcomes, errors = [], 0
    for i in range(1, repeats + 1):
        try:

            r = gate(1, list(ordering), cands, ArbiterBox(str(ds.overlay)),
                     domain=ds.domain)
            blocked = TARGET in (r.blocked or [])
            outcomes.append(blocked)
            print(f"  {i:>3}/{repeats}  {TARGET} {'BLOCKED' if blocked else '*** CLEARED ***'}"
                  f"   (blocked={r.blocked}, pruned={r.pruned})")
        except Exception as e:
            errors += 1
            print(f"  {i:>3}/{repeats}  ERROR {type(e).__name__}: {str(e)[:80]}")

    n = len(outcomes)
    if not n:
        print("\nno rulings obtained")
        return
    blocked_n = sum(outcomes)
    print(f"\n  {blocked_n}/{n} BLOCKED, {n - blocked_n}/{n} CLEARED "
          f"({(n - blocked_n) / n:.0%} clearance)"
          + (f"   [{errors} call(s) errored and are excluded]" if errors else ""))
    print(f"  distribution: {dict(Counter('blocked' if o else 'cleared' for o in outcomes))}")
    print("\n  ⚠️ Report this rate ALONGSIDE the block's cost. 'The gate blocks "
          f"{TARGET}' is not a deterministic property of the system.")

if __name__ == "__main__":
    reps = 11
    for i, a in enumerate(sys.argv):
        if a == "--repeats" and i + 1 < len(sys.argv):
            reps = int(sys.argv[i + 1])
    main(datasets.from_argv(sys.argv).tag, reps)
