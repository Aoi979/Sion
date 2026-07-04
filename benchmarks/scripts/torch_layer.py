#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import time

import torch
import sion  # noqa: F401 - importing loads sion._C and registers torch.ops


def summarize(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "stddev": 0.0}
    if len(values) == 1:
        p90 = values[0]
    else:
        pos = 0.9 * (len(values) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        frac = pos - lo
        p90 = values[lo] * (1.0 - frac) + values[hi] * frac
    mean = statistics.fmean(values)
    variance = statistics.fmean((v - mean) ** 2 for v in values)
    return {
        "min": values[0],
        "max": values[-1],
        "mean": mean,
        "median": statistics.median(values),
        "p90": p90,
        "stddev": math.sqrt(variance),
    }


def parse_shape(shape):
    try:
        return [int(x) for x in shape.split("x")]
    except ValueError as exc:
        raise SystemExit(f"invalid shape: {shape}") from exc


def make_callable(op, shape, api):
    if op == "sgemm":
        if len(shape) != 3:
            raise SystemExit("sgemm shape must be MxNxK")
        m, n, k = shape
        a = torch.zeros((m, k), device="cuda", dtype=torch.float32)
        b = torch.zeros((k, n), device="cuda", dtype=torch.float32)
        work = 2.0 * m * n * k
        if api == "wrapper":
            return lambda: sion.sgemm(a, b), work, "sion.sgemm"
        return lambda: torch.ops.sion.sgemm(a, b, 1.0, 0.0), work, "torch.ops.sion.sgemm"

    if op == "hgemm":
        if len(shape) != 3:
            raise SystemExit("hgemm shape must be MxNxK")
        m, n, k = shape
        a = torch.zeros((m, k), device="cuda", dtype=torch.float16)
        b = torch.zeros((k, n), device="cuda", dtype=torch.float16)
        work = 2.0 * m * n * k
        if api == "wrapper":
            return lambda: sion.hgemm(a, b), work, "sion.hgemm"
        return lambda: torch.ops.sion.hgemm(a, b, 1.0, 0.0), work, "torch.ops.sion.hgemm"

    if op == "flash_attention":
        if len(shape) != 4:
            raise SystemExit("flash_attention shape must be BxHxNxD")
        bsz, heads, seq, dim = shape
        q = torch.zeros((bsz, heads, seq, dim), device="cuda", dtype=torch.float16)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        work = 4.0 * bsz * heads * seq * seq * dim
        if api == "wrapper":
            return lambda: sion.flash_attention(q, k, v), work, "sion.flash_attention"
        return lambda: torch.ops.sion.flash_attention(q, k, v), work, "torch.ops.sion.flash_attention"

    raise SystemExit(f"unsupported op: {op}")


def gpu_elapsed_ms_once(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop))


def choose_iters(fn, args):
    if args.iters > 0:
        return args.iters
    iters = 1
    while iters < args.max_iters:
        if gpu_elapsed_ms_once(fn, iters) >= args.min_sample_ms:
            break
        iters *= 2
    return iters


def run(args):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_device(args.device)
    shape = parse_shape(args.shape)
    fn, work, kernel = make_callable(args.op, shape, args.api)

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    iters = choose_iters(fn, args)
    torch.cuda.synchronize()

    gpu_ms = []
    host_issue_us = []
    e2e_us = []

    for _ in range(args.repeat):
        gpu_ms.append(gpu_elapsed_ms_once(fn, iters) / iters)

    for _ in range(args.repeat):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter()
        host_issue_us.append((t1 - t0) * 1.0e6 / iters)
        torch.cuda.synchronize()

    for _ in range(args.repeat):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        e2e_us.append((t1 - t0) * 1.0e6)

    props = torch.cuda.get_device_properties(args.device)
    result = {
        "op": args.op,
        "layer": "torch",
        "kernel": kernel,
        "shape": args.shape,
        "device": {
            "ordinal": args.device,
            "name": props.name,
            "cc": props.major * 10 + props.minor,
            "sm_count": props.multi_processor_count,
            "max_dynamic_smem": getattr(props, "shared_memory_per_block_optin", 0),
            "max_threads_per_block": getattr(props, "max_threads_per_block", 0),
        },
        "config": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "iters": iters,
            "min_sample_ms": args.min_sample_ms,
        },
        "timing": {
            "gpu_ms": summarize(gpu_ms),
            "host_issue_us": summarize(host_issue_us),
            "e2e_us": summarize(e2e_us),
        },
        "work_units": work,
        "throughput_tunits_per_s": work / (statistics.median(gpu_ms) * 1.0e9) if gpu_ms else 0.0,
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=["sgemm", "hgemm", "flash_attention"])
    parser.add_argument("--shape", required=True)
    parser.add_argument("--api", default="ops", choices=["ops", "wrapper"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=100000)
    parser.add_argument("--min-sample-ms", type=float, default=10.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
