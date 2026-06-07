"""
Generate benchmark plots from results/benchmark.csv
Run after: make run
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

CSV_PATH = "results/benchmark.csv"
OUT_DIR  = "results"

# CPU baselines from Apple M-series (cpp-simd-attention project)
# Hardware: Apple Silicon, 100 GB/s unified memory bandwidth
CPU_BASELINES = {
    256: {
        "CPU Naive":    2.87,
        "CPU NEON SIMD": 31.88,
        "CPU OpenMP":   52.21,
        "CPU Accelerate": 109.3,
    }
}

def load_results():
    df = pd.read_csv(CSV_PATH)
    return df

def plot_gflops(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    naive = df[df["version"] == "naive"]
    tiled = df[df["version"] == "tiled"]

    ax.plot(naive["seq_len"], naive["gflops"], "o--", label="CUDA Naive",  color="#e74c3c", linewidth=2)
    ax.plot(tiled["seq_len"], tiled["gflops"], "o-",  label="CUDA Tiled",  color="#2ecc71", linewidth=2)

    # CPU baselines at seq_len=256 as horizontal reference lines
    colors = ["#95a5a6", "#7f8c8d", "#34495e", "#2c3e50"]
    for (name, val), color in zip(CPU_BASELINES[256].items(), colors):
        ax.axhline(val, linestyle=":", color=color, alpha=0.7, label=f"{name} ({val} GFLOPS/s)")

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("GFLOPS/s", fontsize=12)
    ax.set_title("Attention Kernel Performance: CUDA vs CPU Baselines", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "gflops_comparison.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

def plot_bandwidth_utilization(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    naive = df[df["version"] == "naive"]
    tiled = df[df["version"] == "tiled"]

    ax.plot(naive["seq_len"], naive["bw_util_pct"], "o--", label="CUDA Naive", color="#e74c3c", linewidth=2)
    ax.plot(tiled["seq_len"], tiled["bw_util_pct"], "o-",  label="CUDA Tiled", color="#2ecc71", linewidth=2)

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Memory Bandwidth Utilization (% of A40 peak 696 GB/s)", fontsize=11)
    ax.set_title("Global Memory Bandwidth Utilization", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "bandwidth_utilization.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

def print_summary_table(df):
    print("\n=== Summary Table ===")
    print(f"{'seq_len':>10} {'version':>12} {'time_ms':>10} {'GFLOPS/s':>12} {'BW_util%':>10}")
    print("-" * 58)
    for _, row in df.iterrows():
        print(f"{int(row.seq_len):>10} {row.version:>12} {row.time_ms:>10.3f} "
              f"{row.gflops:>12.1f} {row.bw_util_pct:>10.1f}")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_results()
    print_summary_table(df)
    plot_gflops(df)
    plot_bandwidth_utilization(df)
