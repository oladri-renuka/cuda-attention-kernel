#include "attention.h"
#include <stdio.h>

// ─── QK^T kernel ─────────────────────────────────────────────────────────────
// Thread (row, col) computes one element of the seq_len x seq_len score matrix.
// Every thread independently loads its row of Q and col of K from global DRAM.
// No data reuse — same bytes loaded by multiple threads.

__global__ void qkt_naive_kernel(
    const float* Q, const float* K,
    float* scores,
    int seq_len, int d_k
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= seq_len || col >= seq_len) return;

    float sum = 0.0f;
    for (int i = 0; i < d_k; i++) {
        sum += Q[row * d_k + i] * K[col * d_k + i];
    }
    scores[row * seq_len + col] = sum / sqrtf((float)d_k);
}

// ─── Softmax kernel ───────────────────────────────────────────────────────────
// One thread per row. Numerically stable: subtract max before exp.

__global__ void softmax_kernel(float* scores, int seq_len) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;

    float* row_ptr = scores + row * seq_len;

    float max_val = row_ptr[0];
    for (int i = 1; i < seq_len; i++)
        max_val = fmaxf(max_val, row_ptr[i]);

    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        row_ptr[i] = expf(row_ptr[i] - max_val);
        sum += row_ptr[i];
    }
    for (int i = 0; i < seq_len; i++)
        row_ptr[i] /= sum;
}

// ─── scores × V kernel ───────────────────────────────────────────────────────
// Thread (row, col) computes one element of the output matrix.
// Same naive global-memory pattern as qkt_naive_kernel.

__global__ void scores_v_naive_kernel(
    const float* scores, const float* V,
    float* output,
    int seq_len, int d_k
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= seq_len || col >= d_k) return;

    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++)
        sum += scores[row * seq_len + i] * V[i * d_k + col];
    output[row * d_k + col] = sum;
}

// ─── Host launcher ────────────────────────────────────────────────────────────

void run_attention_naive(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_scores, float* d_out,
    int seq_len, int d_k
) {
    dim3 block(16, 16);
    dim3 grid_sq((seq_len + 15) / 16, (seq_len + 15) / 16);
    dim3 grid_out((seq_len + 15) / 16, (d_k + 15) / 16);

    qkt_naive_kernel<<<grid_sq, block>>>(d_Q, d_K, d_scores, seq_len, d_k);
    softmax_kernel<<<(seq_len + 255) / 256, 256>>>(d_scores, seq_len);
    scores_v_naive_kernel<<<grid_out, block>>>(d_scores, d_V, d_out, seq_len, d_k);
}
