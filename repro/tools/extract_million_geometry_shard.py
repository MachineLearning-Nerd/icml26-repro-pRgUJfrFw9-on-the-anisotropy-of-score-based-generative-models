"""Extract a compact, hash-addressed geometry shard record from an orx log."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys


MARKER = "CLAIM_1_4_MILLION_SHARD="


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    result = None
    for line in sys.stdin:
        if line.startswith(MARKER):
            result = json.loads(line[len(MARKER) :])
    if result is None:
        raise RuntimeError(f"{MARKER} record not found on stdin")

    compact = {
        "source_run_id": args.run_id,
        "source_commit": args.commit,
        "source_result_sha256": hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "scientific_status": result["scientific_status"],
        "claim_verdict": result["claim_verdict"],
        "design": result["design"],
        "author_probe": result["author_probe"],
        "compute": result["compute"],
        "network_accounting": result["network_accounting"],
        "matrix_audit": result["matrix_audit"],
        "negative_control": result["negative_control"],
        "structural_pass": result["structural_pass"],
        "complete_geometry": result["complete_geometry"],
        "parity_half_geometries": result["parity_half_geometries"],
        "runtime": result["runtime"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
