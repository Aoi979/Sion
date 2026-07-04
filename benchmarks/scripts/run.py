#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TORCH_PYTHON = Path("/home/aoi211/.conda/envs/torch/bin/python")


def load_case(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    defaults = data.get("defaults", {})
    jobs = data.get("jobs", [])
    return defaults, jobs


def layer_list(value):
    if isinstance(value, list):
        return value
    return [x.strip() for x in str(value).split(",") if x.strip()]


def build_env(build_dir):
    env = os.environ.copy()
    python_path = str(ROOT / "python")
    env["PYTHONPATH"] = python_path + os.pathsep + env.get("PYTHONPATH", "")
    ld_parts = [
        str(build_dir / "src" / "sion"),
        str(build_dir / "src" / "felix"),
    ]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts + [env.get("LD_LIBRARY_PATH", "")])
    return env


def checked_subprocess(cmd, **kwargs):
    proc = subprocess.run(cmd, text=True, capture_output=True, **kwargs)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc


def run_native(binary, job, layer, cfg):
    cmd = [
        str(binary),
        "--op",
        job["op"],
        "--layer",
        layer,
        "--shape",
        job["shape"],
        "--kernel",
        job.get("kernel", "auto"),
        "--warmup",
        str(cfg["warmup"]),
        "--repeat",
        str(cfg["repeat"]),
        "--iters",
        str(cfg.get("iters", 0)),
        "--min-sample-ms",
        str(cfg["min_sample_ms"]),
    ]
    proc = checked_subprocess(cmd)
    return json.loads(proc.stdout)


def run_torch(job, cfg, build_dir, python):
    cmd = [
        str(python),
        str(ROOT / "benchmarks" / "scripts" / "torch_layer.py"),
        "--op",
        job["op"],
        "--shape",
        job["shape"],
        "--api",
        job.get("api", "ops"),
        "--warmup",
        str(cfg["warmup"]),
        "--repeat",
        str(cfg["repeat"]),
        "--iters",
        str(cfg.get("iters", 0)),
        "--min-sample-ms",
        str(cfg["min_sample_ms"]),
    ]
    proc = checked_subprocess(cmd, env=build_env(build_dir))
    return json.loads(proc.stdout)


def normalize_cfg(defaults, job, args):
    cfg = {
        "warmup": args.warmup if args.warmup is not None else defaults.get("warmup", 10),
        "repeat": args.repeat if args.repeat is not None else defaults.get("repeat", 30),
        "iters": args.iters if args.iters is not None else defaults.get("iters", 0),
        "min_sample_ms": args.min_sample_ms
        if args.min_sample_ms is not None
        else defaults.get("min_sample_ms", 10.0),
    }
    cfg.update(job.get("timing", {}))
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path)
    parser.add_argument("--op")
    parser.add_argument("--shape")
    parser.add_argument("--kernel", default="auto")
    parser.add_argument("--layers", default="felix")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-pytorch")
    parser.add_argument("--binary", type=Path)
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_TORCH_PYTHON if DEFAULT_TORCH_PYTHON.exists() else Path(sys.executable),
        help="Python executable used for torch-layer measurements",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--min-sample-ms", type=float)
    args = parser.parse_args()

    if args.case:
        defaults, jobs = load_case(args.case)
    else:
        if not args.op or not args.shape:
            parser.error("either --case or both --op/--shape are required")
        defaults = {}
        jobs = [
            {
                "op": args.op,
                "shape": args.shape,
                "kernel": args.kernel,
                "layers": layer_list(args.layers),
            }
        ]

    binary = args.binary or args.build_dir / "benchmarks" / "sion_bench"
    results = []
    for job in jobs:
        cfg = normalize_cfg(defaults, job, args)
        for layer in layer_list(job.get("layers", args.layers)):
            if layer in ("raw", "felix"):
                results.append(run_native(binary, job, layer, cfg))
            elif layer == "torch":
                results.append(run_torch(job, cfg, args.build_dir, args.python))
            else:
                raise SystemExit(f"unknown layer: {layer}")

    payload = json.dumps(results, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
