"""Step 8: charging-protocol metadata of the MATR-1 cells (needs the raw pickles).

All MATR-1 cells use a two-step fast charge: C1 from 0 % to Q1 % SOC, then C2
to 80 % SOC (then 1C-CV to full, identical for all cells). These three numbers
are fixed before cycling starts, so they are *design variables*, not
degradation measurements.

Output: results/tables/protocols.csv (cell_id, partition, c1, soc_switch, c2,
protocol, protocol_seen_in_train)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.io import load_cell, resolve_raw_dir  # noqa: E402
from batteryrul.splits import read_manifests  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw_dir = resolve_raw_dir(args.data_dir, cfg["data"]["raw_dir"])
    train, test, _ = read_manifests(repo_path(cfg["split"]["manifest_dir"]))
    rows = []
    for part, cells in (("train", train), ("test", test)):
        for c in cells:
            steps = load_cell(raw_dir, c)["charge_protocol"]
            assert len(steps) == 2, (c, steps)
            rows.append({"cell_id": c, "partition": part, "c1": steps[0]["rate_in_C"],
                         "soc_switch": steps[0]["end_soc"], "c2": steps[1]["rate_in_C"]})
    df = pd.DataFrame(rows)
    df["protocol"] = [f"{a:g}C({s:g}%)-{b:g}C" for a, s, b in zip(df.c1, df.soc_switch, df.c2)]
    seen = set(df.loc[df.partition == "train", "protocol"])
    df["protocol_seen_in_train"] = df.protocol.isin(seen) & (df.partition == "test")
    out = repo_path(cfg["output"]["results_dir"]) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "protocols.csv", index=False)
    print(f"{df.protocol.nunique()} protocols over {len(df)} cells; "
          f"{int(df.protocol_seen_in_train.sum())}/{len(test)} test cells share a protocol with a training cell")


if __name__ == "__main__":
    main()
