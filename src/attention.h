#pragma once
#include <cuda_runtime.h>
#include <math.h>

// ─── Naive kernels ───────────────────────────────────────────────────────────

__global__ void qkt_naive_kernel(
    const float* Q, const float* K,
    float* scores,
    int seq_len, int d_k
);

__global__ void softmax_kernel(float* scores, int seq_len);

__global__ void scores_v_naive_kernel(
    const float* scores, const float* V,
    float* output,
    int seq_len, int d_k
);

// ─── Tiled kernels ───────────────────────────────────────────────────────────

#define TILE_SIZE 32

__global__ void qkt_tiled_kernel(
    const float* Q, const float* K,
    float* scores,
    int seq_len, int d_k
);

__global__ void scores_v_tiled_kernel(
    const float* scores, const float* V,
    float* output,
    int seq_len, int d_k
);

// ─── Host launcher declarations ──────────────────────────────────────────────

void run_attention_naive(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_scores, float* d_out,
    int seq_len, int d_k
);

void run_attention_tiled(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_scores, float* d_out,
    int seq_len, int d_k
);
