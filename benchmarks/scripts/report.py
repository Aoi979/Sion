#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_results(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()

    rows = load_results(args.result)
    print("| op | layer | kernel | shape | gpu median ms | host issue median us | e2e median us | throughput Tunit/s |")
    print("|----|-------|--------|-------|---------------|----------------------|---------------|--------------------|")
    for r in rows:
        timing = r["timing"]
        print(
            "| {op} | {layer} | `{kernel}` | {shape} | {gpu:.6g} | {host:.6g} | {e2e:.6g} | {thr:.6g} |".format(
                op=r["op"],
                layer=r["layer"],
                kernel=r["kernel"],
                shape=r["shape"],
                gpu=timing["gpu_ms"]["median"],
                host=timing["host_issue_us"]["median"],
                e2e=timing["e2e_us"]["median"],
                thr=r.get("throughput_tunits_per_s", 0.0),
            )
        )


if __name__ == "__main__":
    main()
