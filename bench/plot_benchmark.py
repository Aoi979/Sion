#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import os

# -----------------------------
# 函数：解析 Markdown 表格
# -----------------------------
def parse_markdown_table(md_text):
    lines = md_text.strip().splitlines()
    data_lines = [line for line in lines if line.startswith('|')]
    
    # 去掉分隔行（含 --- 的行）
    data_lines = [line for line in data_lines if not re.match(r'\|\s*-+', line)]
    
    if not data_lines:
        raise ValueError("Markdown 表格为空或格式不对")

    # 提取表头
    header = [h.strip() for h in data_lines[0].split('|')[1:-1]]
    header = [re.sub(r'\(.*?\)', '', h).strip() for h in header]

    # 提取每行数据
    rows = []
    for line in data_lines[1:]:
        cols = [c.strip() for c in line.split('|')[1:-1]]
        rows.append(cols)
    
    # 转为 DataFrame
    df = pd.DataFrame(rows, columns=header)
    
    # 尝试将数值列转换为 float
    for col in df.columns:
        if col.lower() in ['shape', 'winner']:
            continue
        df[col] = df[col].str.replace('x', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def detect_ref_label(md_text):
    # 先尝试匹配标题中的 "(ref: xxx)"
    m = re.search(r'\(ref:\s*([^)]+)\)', md_text, flags=re.IGNORECASE)
    if not m:
        # 再尝试匹配列表中的 "- ref: xxx"
        m = re.search(r'^\s*-\s*ref:\s*([^\n\r]+)', md_text,
                      flags=re.IGNORECASE | re.MULTILINE)

    raw = m.group(1).strip() if m else "cuBLAS"
    key = raw.lower()
    mapping = {
        "cublas": "cuBLAS",
        "libtorch_sdpa": "Torch SDPA",
        "torch_sdpa": "Torch SDPA",
        "torch": "Torch",
        "none": "Ref",
    }
    return mapping.get(key, raw)


def find_col(df, exact_candidates=None, regex_candidates=None, exclude=None):
    exact_candidates = exact_candidates or []
    regex_candidates = regex_candidates or []
    exclude = set(exclude or [])

    for name in exact_candidates:
        if name in df.columns and name not in exclude:
            return name

    for pat in regex_candidates:
        rgx = re.compile(pat, flags=re.IGNORECASE)
        for col in df.columns:
            if col in exclude:
                continue
            if rgx.search(col):
                return col
    return None

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GEMM Markdown results")
    parser.add_argument("md_file", help="Markdown file containing the GEMM table")
    args = parser.parse_args()

    # Markdown 文件所在目录
    md_dir = os.path.dirname(os.path.abspath(args.md_file))
    md_basename = os.path.splitext(os.path.basename(args.md_file))[0]

    # 读取 Markdown 文件
    with open(args.md_file, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    # 解析表格
    df = parse_markdown_table(md_text)
    ref_label = detect_ref_label(md_text)
    print("[INFO] 列名:", df.columns.tolist())
    print("[INFO] ref 标签:", ref_label)

    shape_col = find_col(df, exact_candidates=["Shape"],
                         regex_candidates=[r'^shape'])
    sion_avg_col = find_col(df, exact_candidates=["Sion avg_ms"],
                            regex_candidates=[r'^sion.*avg_ms'])
    ref_avg_col = find_col(
        df,
        exact_candidates=["cuBLAS avg_ms", "Ref avg_ms"],
        regex_candidates=[r'^(?!sion).*(cublas|ref).*avg_ms'],
        exclude=[sion_avg_col] if sion_avg_col else None
    )

    sion_tflops_col = find_col(
        df,
        exact_candidates=["Sion TFLOPS", "Sion eff_tflops"],
        regex_candidates=[r'^sion.*(tflops|eff_tflops)$']
    )
    ref_tflops_col = find_col(
        df,
        exact_candidates=["cuBLAS TFLOPS", "Ref TFLOPS", "Ref eff_tflops"],
        regex_candidates=[r'^(?!sion).*(cublas|ref).*(tflops|eff_tflops)$'],
        exclude=[sion_tflops_col] if sion_tflops_col else None
    )
    ratio_col = find_col(
        df,
        exact_candidates=["Sion TFLOPS ratio", "Sion eff_tflops ratio"],
        regex_candidates=[r'^sion.*(tflops|eff_tflops).*ratio']
    )

    if shape_col is None or sion_avg_col is None or sion_tflops_col is None:
        raise ValueError("未找到必需列: Shape / Sion avg_ms / Sion TFLOPS")
    
    # 设置 seaborn 样式
    sns.set(style="whitegrid")
    x = df[shape_col]
    has_ref_avg = ref_avg_col is not None and df[ref_avg_col].notna().any()
    has_ref_tflops = ref_tflops_col is not None and df[ref_tflops_col].notna().any()

    # -----------------------------
    # 1️⃣ TFLOPS 对比柱状图
    # -----------------------------
    plt.figure(figsize=(12,6))
    plt.bar(x, df[sion_tflops_col], width=0.4, label='Sion', align='edge')
    if has_ref_tflops:
        plt.bar(x, df[ref_tflops_col], width=-0.4, label=ref_label, align='edge')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("TFLOPS")
    plt.title(f"Sion vs {ref_label} TFLOPS")
    plt.legend()
    plt.tight_layout()
    tfops_file = os.path.join(md_dir, f"{md_basename}_tfops_comparison.png")
    plt.savefig(tfops_file, dpi=200)
    plt.show()
    
    # -----------------------------
    # 2️⃣ Speedup 折线图
    # -----------------------------
    speedup_series = None
    if ratio_col is not None and df[ratio_col].notna().any():
        speedup_series = df[ratio_col]
    elif has_ref_tflops:
        speedup_series = df[sion_tflops_col] / df[ref_tflops_col]
    elif has_ref_avg:
        speedup_series = df[ref_avg_col] / df[sion_avg_col]

    plt.figure(figsize=(12,6))
    if speedup_series is not None:
        plt.plot(x, speedup_series, marker='o', color='orange',
                 label=f'Sion / {ref_label}')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(f"Sion / {ref_label} ratio")
        plt.title(f"Sion Speedup vs {ref_label}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        speedup_file = os.path.join(md_dir, f"{md_basename}_speedup_ratio.png")
        plt.savefig(speedup_file, dpi=200)
        plt.show()
    else:
        speedup_file = os.path.join(md_dir, f"{md_basename}_speedup_ratio.png")
        print("[WARN] 无法计算 speedup（缺少 ratio/ref 列），跳过该图")
    
    # -----------------------------
    # 3️⃣ avg_ms 对比柱状图
    # -----------------------------
    plt.figure(figsize=(12,6))
    plt.bar(x, df[sion_avg_col], width=0.4, label='Sion', align='edge')
    if has_ref_avg:
        plt.bar(x, df[ref_avg_col], width=-0.4, label=ref_label, align='edge')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("avg_ms")
    plt.title(f"Sion vs {ref_label} avg_ms")
    plt.legend()
    plt.tight_layout()
    avgms_file = os.path.join(md_dir, f"{md_basename}_avg_ms_comparison.png")
    plt.savefig(avgms_file, dpi=200)
    plt.show()
    
    print(f"[INFO] 生成完毕：{tfops_file}, {speedup_file}, {avgms_file}")
