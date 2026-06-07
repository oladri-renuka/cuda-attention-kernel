#include "attention.h"
#include <stdio.h>

// ─── Tiled QK^T kernel ────────────────────────────────────────────────────────
// Divides the d_k dimension into TILE_SIZE chunks.
// All threads in a block cooperatively load one tile of Q and one tile of K
// into shared memory, then compute partial dot products using on-chip SRAM
// (~4 cycle latency vs ~300 cycles for global DRAM).
//
// Global memory traffic reduction: each element is loaded once into shared
// memory and reused TILE_SIZE times. For TILE_SIZE=32, this is a 32x reduction
// in global memory reads vs the naive kernel.
//
// Shared memory used per block: 2 * TILE_SIZE * TILE_SIZE * 4 bytes = 8KB
// A40 has 48KB shared memory per SM → fits ~6 concurrent blocks per SM.

__global__ void qkt_tiled_kernel(
    const float* Q, const float* K,
    float* scores,
    int seq_len, int d_k
) {
    __shared__ float Q_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float K_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    float sum = 0.0f;

    int num_tiles = (d_k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {
        int k_idx = tile * TILE_SIZE + threadIdx.y;

        Q_tile[threadIdx.x][threadIdx.y] = (row < seq_len && k_idx < d_k)
            ? Q[row * d_k + k_idx] : 0.0f;

        int k_col = blockIdx.y * TILE_SIZE + threadIdx.x;
        int k_row = tile * TILE_SIZE + threadIdx.y;
        K_tile[threadIdx.x][threadIdx.y] = (k_col < seq_len && k_row < d_k)
            ? K[k_col * d_k + k_row] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
            sum += Q_tile[threadIdx.x][i] * K_tile[threadIdx.y][i];

        __syncthreads();
    }

    if (row < seq_len && col < seq_len)
        scores[row * seq_len + col] = sum / sqrtf((float)d_k);
}

// ─── Tiled scores x V kernel ─────────────────────────────────────────────────

__global__ void scores_v_tiled_kernel(
    const float* scores, const float* V,
    float* output,
    int seq_len, int d_k
) {
    __shared__ float S_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float V_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    float sum = 0.0f;

    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {
        int k_idx = tile * TILE_SIZE + threadIdx.y;

        S_tile[threadIdx.x][threadIdx.y] = (row < seq_len && k_idx < seq_len)
            ? scores[row * seq_len + k_idx] : 0.0f;

        int v_col = blockIdx.y * TILE_SIZE + threadIdx.x;
        int v_row = tile * TILE_SIZE + threadIdx.y;
        V_tile[threadIdx.x][threadIdx.y] = (v_col < d_k && v_row < seq_len)
            ? V[v_row * d_k + v_col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
            sum += S_tile[threadIdx.x][i] * V_tile[threadIdx.y][i];

        __syncthreads();
    }

    if (row < seq_len && col < d_k)
        output[row * d_k + col] = sum;
}

// ─── Host launcher ────────────────────────────────────────────────────────────

void run_attention_tiled(
    const float* d_Q, const float* d_K, const float* d_V,
    float* d_scores, float* d_out,
    int seq_len, int d_k
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid_sq((seq_len + TILE_SIZE - 1) / TILE_SIZE,
                 (seq_len + TILE_SIZE - 1) / TILE_SIZE);
    dim3 grid_out((seq_len + TILE_SIZE - 1) / TILE_SIZE,
                  (d_k    + TILE_SIZE - 1) / TILE_SIZE);

    qkt_tiled_kernel<<<grid_sq, block>>>(d_Q, d_K, d_scores, seq_len, d_k);
    softmax_kernel<<<(seq_len + 255) / 256, 256>>>(d_scores, seq_len);
    scores_v_tiled_kernel<<<grid_out, block>>>(d_scores, d_V, d_out, seq_len, d_k);
}
