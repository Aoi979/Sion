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
    print("[INFO] 列名:", df.columns.tolist())
    
    # 设置 seaborn 样式
    sns.set(style="whitegrid")
    x = df['Shape']

    # -----------------------------
    # 1️⃣ TFLOPS 对比柱状图
    # -----------------------------
    plt.figure(figsize=(12,6))
    plt.bar(x, df['Sion TFLOPS'], width=0.4, label='Sion', align='edge')
    plt.bar(x, df['cuBLAS TFLOPS'], width=-0.4, label='cuBLAS', align='edge')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("TFLOPS")
    plt.title("Sion vs cuBLAS TFLOPS")
    plt.legend()
    plt.tight_layout()
    tfops_file = os.path.join(md_dir, f"{md_basename}_tfops_comparison.png")
    plt.savefig(tfops_file, dpi=200)
    plt.show()
    
    # -----------------------------
    # 2️⃣ Speedup 折线图
    # -----------------------------
    plt.figure(figsize=(12,6))
    plt.plot(x, df['Sion TFLOPS ratio'], marker='o', color='orange', label='Sion / cuBLAS')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Sion / cuBLAS TFLOPS ratio")
    plt.title("Sion Speedup vs cuBLAS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    speedup_file = os.path.join(md_dir, f"{md_basename}_speedup_ratio.png")
    plt.savefig(speedup_file, dpi=200)
    plt.show()
    
    # -----------------------------
    # 3️⃣ avg_ms 对比柱状图
    # -----------------------------
    plt.figure(figsize=(12,6))
    plt.bar(x, df['Sion avg_ms'], width=0.4, label='Sion', align='edge')
    plt.bar(x, df['cuBLAS avg_ms'], width=-0.4, label='cuBLAS', align='edge')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("avg_ms")
    plt.title("Sion vs cuBLAS avg_ms")
    plt.legend()
    plt.tight_layout()
    avgms_file = os.path.join(md_dir, f"{md_basename}_avg_ms_comparison.png")
    plt.savefig(avgms_file, dpi=200)
    plt.show()
    
    print(f"[INFO] 生成完毕：{tfops_file}, {speedup_file}, {avgms_file}")