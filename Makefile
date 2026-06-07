NVCC        = nvcc
CUDA_FLAGS  = -O2 -arch=sm_80 -std=c++17   # sm_80 = A40
CUDA_FLAGS += -Xcompiler -Wall
INCLUDES    = -I src/

# ─── Targets ─────────────────────────────────────────────────────────────────

.PHONY: all bench clean test

all: bench

bench: benchmarks/bench_cuda.cu \
       src/attention_cuda_naive.cu \
       src/attention_cuda_tiled.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) \
		src/attention_cuda_naive.cu \
		src/attention_cuda_tiled.cu \
		benchmarks/bench_cuda.cu \
		-o bench
	@echo "Build OK — run with: ./bench"

run: bench
	mkdir -p results
	./bench

# Profile with Nsight (requires sudo or appropriate permissions)
profile: bench
	ncu --metrics sm__throughput.avg_pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
dram__bytes.sum \
	--target-processes all \
	./bench

clean:
	rm -f bench
	rm -f results/benchmark.csv
