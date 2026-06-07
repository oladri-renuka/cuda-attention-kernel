#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../src/attention.h"

// ─── Utility ─────────────────────────────────────────────────────────────────

void rand_fill(float* arr, int n) {
    for (int i = 0; i < n; i++)
        arr[i] = (float)rand() / RAND_MAX - 0.5f;
}

// Check two output matrices agree within tolerance (correctness test)
int outputs_match(const float* a, const float* b, int n, float tol) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tol) {
            printf("  MISMATCH at index %d: naive=%.6f tiled=%.6f diff=%.6f\n",
                   i, a[i], b[i], fabsf(a[i] - b[i]));
            return 0;
        }
    }
    return 1;
}

// ─── Single benchmark run ─────────────────────────────────────────────────────

typedef struct {
    float avg_ms;
    float gflops;
    float bandwidth_gb;
} BenchResult;

BenchResult benchmark(
    int seq_len, int d_k, int is_tiled, int warmup, int iters
) {
    size_t mat_size   = (size_t)seq_len * d_k    * sizeof(float);
    size_t score_size = (size_t)seq_len * seq_len * sizeof(float);

    // Host alloc + init
    float *h_Q = (float*)malloc(mat_size);
    float *h_K = (float*)malloc(mat_size);
    float *h_V = (float*)malloc(mat_size);
    rand_fill(h_Q, seq_len * d_k);
    rand_fill(h_K, seq_len * d_k);
    rand_fill(h_V, seq_len * d_k);

    // Device alloc
    float *d_Q, *d_K, *d_V, *d_scores, *d_out;
    cudaMalloc(&d_Q,      mat_size);
    cudaMalloc(&d_K,      mat_size);
    cudaMalloc(&d_V,      mat_size);
    cudaMalloc(&d_scores, score_size);
    cudaMalloc(&d_out,    mat_size);

    cudaMemcpy(d_Q, h_Q, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, mat_size, cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        if (is_tiled)
            run_attention_tiled(d_Q, d_K, d_V, d_scores, d_out, seq_len, d_k);
        else
            run_attention_naive(d_Q, d_K, d_V, d_scores, d_out, seq_len, d_k);
    }
    cudaDeviceSynchronize();

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        if (is_tiled)
            run_attention_tiled(d_Q, d_K, d_V, d_scores, d_out, seq_len, d_k);
        else
            run_attention_naive(d_Q, d_K, d_V, d_scores, d_out, seq_len, d_k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / iters;

    // FLOPs: QK^T (2 * seq^2 * d_k) + scores*V (2 * seq^2 * d_k)
    long long flops = 4LL * seq_len * seq_len * d_k;
    float gflops = (float)flops / (avg_ms * 1e6f);

    // Bytes moved (naive lower bound): Q+K+scores read for QK^T, scores+V+out for scores*V
    long long bytes = 2LL * (
        (long long)seq_len * d_k * 2 +      // Q + K
        (long long)seq_len * seq_len +       // scores write
        (long long)seq_len * seq_len +       // scores read
        (long long)seq_len * d_k +           // V
        (long long)seq_len * d_k             // output write
    ) * sizeof(float);
    float bandwidth_gb = (float)bytes / (avg_ms * 1e6f);

    BenchResult r = { avg_ms, gflops, bandwidth_gb };

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_scores); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_Q); free(h_K); free(h_V);

    return r;
}

// ─── Correctness check ────────────────────────────────────────────────────────

void correctness_check(int seq_len, int d_k) {
    size_t mat_size   = (size_t)seq_len * d_k    * sizeof(float);
    size_t score_size = (size_t)seq_len * seq_len * sizeof(float);

    float *h_Q = (float*)malloc(mat_size);
    float *h_K = (float*)malloc(mat_size);
    float *h_V = (float*)malloc(mat_size);
    rand_fill(h_Q, seq_len * d_k);
    rand_fill(h_K, seq_len * d_k);
    rand_fill(h_V, seq_len * d_k);

    float *d_Q, *d_K, *d_V;
    float *d_scores_naive, *d_out_naive;
    float *d_scores_tiled, *d_out_tiled;

    cudaMalloc(&d_Q, mat_size);
    cudaMalloc(&d_K, mat_size);
    cudaMalloc(&d_V, mat_size);
    cudaMalloc(&d_scores_naive, score_size);
    cudaMalloc(&d_out_naive,    mat_size);
    cudaMalloc(&d_scores_tiled, score_size);
    cudaMalloc(&d_out_tiled,    mat_size);

    cudaMemcpy(d_Q, h_Q, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, mat_size, cudaMemcpyHostToDevice);

    run_attention_naive(d_Q, d_K, d_V, d_scores_naive, d_out_naive, seq_len, d_k);
    run_attention_tiled(d_Q, d_K, d_V, d_scores_tiled, d_out_tiled, seq_len, d_k);
    cudaDeviceSynchronize();

    float *h_out_naive = (float*)malloc(mat_size);
    float *h_out_tiled = (float*)malloc(mat_size);
    cudaMemcpy(h_out_naive, d_out_naive, mat_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_tiled, d_out_tiled, mat_size, cudaMemcpyDeviceToHost);

    int ok = outputs_match(h_out_naive, h_out_tiled, seq_len * d_k, 1e-3f);
    printf("Correctness check (seq=%d, d_k=%d): %s\n",
           seq_len, d_k, ok ? "PASS" : "FAIL");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_scores_naive); cudaFree(d_out_naive);
    cudaFree(d_scores_tiled); cudaFree(d_out_tiled);
    free(h_Q); free(h_K); free(h_V);
    free(h_out_naive); free(h_out_tiled);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    srand(42);

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d | Shared mem/SM: %zu KB | Global mem: %zu MB\n\n",
           prop.multiProcessorCount,
           prop.sharedMemPerBlock / 1024,
           prop.totalGlobalMem / (1024 * 1024));

    // Correctness
    printf("=== Correctness ===\n");
    correctness_check(64,  64);
    correctness_check(256, 64);
    correctness_check(512, 64);
    printf("\n");

    // Benchmark table
    int seq_lens[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int d_k = 128;
    int n_seq = sizeof(seq_lens) / sizeof(seq_lens[0]);

    // A40 peak bandwidth for utilization calculation (GB/s)
    float peak_bw = 696.0f;

    printf("=== Benchmark (d_k=%d, warmup=5, iters=100) ===\n", d_k);
    printf("%-10s %-12s %-10s %-12s %-12s %-12s\n",
           "seq_len", "version", "time_ms", "GFLOPS/s", "BW_GB/s", "BW_util%");
    printf("%-10s %-12s %-10s %-12s %-12s %-12s\n",
           "-------", "-------", "-------", "--------", "-------", "--------");

    // CSV output
    FILE* csv = fopen("results/benchmark.csv", "w");
    fprintf(csv, "seq_len,version,time_ms,gflops,bandwidth_gb,bw_util_pct\n");

    for (int s = 0; s < n_seq; s++) {
        int seq = seq_lens[s];

        BenchResult naive = benchmark(seq, d_k, 0, 10, 1000);
        BenchResult tiled = benchmark(seq, d_k, 1, 10, 1000);

        float speedup = naive.avg_ms / tiled.avg_ms;

        printf("%-10d %-12s %-10.3f %-12.1f %-12.1f %-12.1f\n",
               seq, "naive", naive.avg_ms, naive.gflops,
               naive.bandwidth_gb, naive.bandwidth_gb / peak_bw * 100.0f);
        printf("%-10d %-12s %-10.3f %-12.1f %-12.1f %-12.1f  (%.1fx over naive)\n",
               seq, "tiled", tiled.avg_ms, tiled.gflops,
               tiled.bandwidth_gb, tiled.bandwidth_gb / peak_bw * 100.0f, speedup);
        printf("\n");

        fprintf(csv, "%d,naive,%.4f,%.2f,%.2f,%.2f\n",
                seq, naive.avg_ms, naive.gflops,
                naive.bandwidth_gb, naive.bandwidth_gb / peak_bw * 100.0f);
        fprintf(csv, "%d,tiled,%.4f,%.2f,%.2f,%.2f\n",
                seq, tiled.avg_ms, tiled.gflops,
                tiled.bandwidth_gb, tiled.bandwidth_gb / peak_bw * 100.0f);
    }

    fclose(csv);
    printf("Results written to results/benchmark.csv\n");
    return 0;
}
