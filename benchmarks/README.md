# Sion Benchmarks

The benchmark stack separates three timing layers:

- `raw`: direct low-level CUDA launch for kernels that expose a prepared path.
- `felix`: public Felix launch API, including dispatch and launcher glue.
- `torch`: `torch.ops.sion.*` or Python wrapper entry points.

Metrics are intentionally separate:

- `gpu_ms`: CUDA event timing around the measured callable.
- `host_issue_us`: CPU time spent issuing launches, with synchronization after the sample.
- `e2e_us`: one call plus synchronization.

Build:

```bash
cmake --build build-pytorch --target sion_bench -- -j2
```

Run one native case:

```bash
build-pytorch/benchmarks/sion_bench \
  --op flash_attention \
  --layer raw \
  --shape 1x1x128x64 \
  --kernel sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2
```

Run a sweep, including the PyTorch layer:

```bash
python benchmarks/scripts/run.py \
  --case benchmarks/cases/flash_attention_v2.json \
  --out benchmarks/results/fa_v2.json
```

Render a compact markdown table:

```bash
python benchmarks/scripts/report.py benchmarks/results/fa_v2.json
```
