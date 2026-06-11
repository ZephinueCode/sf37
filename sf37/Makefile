CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
LDLIBS ?= -lm -pthread

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
CUDA_SPARK_ARCH ?= sm_121
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas

.PHONY: all help clean cpu cuda cuda-spark cuda-generic cuda-test inspect test

all: help

help:
	@echo "SF37 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10 (CUDA_SPARK_ARCH=$(CUDA_SPARK_ARCH))"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cuda-test           Build and run CUDA op parity tests"
	@echo "  make cpu                 Build CPU-only ./sf37, ./sf37-bench, ./sf37-server"
	@echo "  make inspect             Build CPU and inspect the local SF37 GGUF"
	@echo "  make test                Build and run SF37 layout tests"
	@echo "  make clean               Remove build outputs"

cpu: sf37_cli_cpu.o sf37_bench_cpu.o sf37_server_cpu.o sf37_kvstore_cpu.o rax_cpu.o sf37_cpu.o sf37_quant_cpu.o sf37_ops_cpu.o sf37_image_cpu.o
	$(CC) $(CFLAGS) -o sf37 sf37_cli_cpu.o sf37_cpu.o sf37_quant_cpu.o sf37_ops_cpu.o sf37_image_cpu.o $(LDLIBS)
	$(CC) $(CFLAGS) -o sf37-bench sf37_bench_cpu.o sf37_cpu.o sf37_quant_cpu.o sf37_ops_cpu.o $(LDLIBS)
	$(CC) $(CFLAGS) -o sf37-server sf37_server_cpu.o sf37_kvstore_cpu.o rax_cpu.o sf37_cpu.o sf37_quant_cpu.o sf37_ops_cpu.o sf37_image_cpu.o $(LDLIBS)

cuda-spark:
	$(MAKE) -B sf37 sf37-bench sf37-server CUDA_ARCH="$(CUDA_SPARK_ARCH)"

cuda-generic:
	$(MAKE) -B sf37 sf37-bench sf37-server CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_121"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B sf37 sf37-bench sf37-server CUDA_ARCH="$(CUDA_ARCH)"

sf37: sf37_cli.o sf37.o sf37_quant.o sf37_ops.o sf37_image.o sf37_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

sf37-bench: sf37_bench.o sf37.o sf37_quant.o sf37_ops.o sf37_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

sf37-server: sf37_server.o sf37_kvstore.o rax.o sf37.o sf37_quant.o sf37_ops.o sf37_image.o sf37_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

sf37_cli_cpu.o: sf37_cli.c sf37.h sf37_image.h
	$(CC) $(CFLAGS) -DSF37_CPU_ONLY -c -o $@ sf37_cli.c

sf37_bench_cpu.o: sf37_bench.c sf37.h
	$(CC) $(CFLAGS) -DSF37_CPU_ONLY -c -o $@ sf37_bench.c

sf37_server_cpu.o: sf37_server.c sf37.h sf37_image.h sf37_kvstore.h rax.h
	$(CC) $(CFLAGS) -DSF37_CPU_ONLY -c -o $@ sf37_server.c

sf37_kvstore_cpu.o: sf37_kvstore.c sf37_kvstore.h sf37.h
	$(CC) $(CFLAGS) -DSF37_CPU_ONLY -c -o $@ sf37_kvstore.c

rax_cpu.o: rax.c rax.h
	$(CC) $(CFLAGS) -DSF37_CPU_ONLY -c -o $@ rax.c

sf37_cpu.o: sf37.c sf37.h sf37_ops.h sf37_quant.h
	$(CC) $(CFLAGS) -DSF37_CPU_ONLY -c -o $@ sf37.c

sf37_quant_cpu.o: sf37_quant.c sf37_quant.h
	$(CC) $(CFLAGS) -DSF37_CPU_ONLY -c -o $@ sf37_quant.c

sf37_ops_cpu.o: sf37_ops.c sf37_ops.h sf37_quant.h
	$(CC) $(CFLAGS) -DSF37_CPU_ONLY -c -o $@ sf37_ops.c

sf37_image_cpu.o: sf37_image.c sf37_image.h sf37.h third_party/stb_image.h
	$(CC) $(CFLAGS) -DSF37_CPU_ONLY -c -o $@ sf37_image.c

sf37_cli.o: sf37_cli.c sf37.h sf37_image.h
	$(CC) $(CFLAGS) -DSF37_USE_CUDA -c -o $@ sf37_cli.c

sf37_bench.o: sf37_bench.c sf37.h
	$(CC) $(CFLAGS) -DSF37_USE_CUDA -c -o $@ sf37_bench.c

sf37_server.o: sf37_server.c sf37.h sf37_image.h sf37_kvstore.h rax.h
	$(CC) $(CFLAGS) -DSF37_USE_CUDA -c -o $@ sf37_server.c

sf37_kvstore.o: sf37_kvstore.c sf37_kvstore.h sf37.h
	$(CC) $(CFLAGS) -DSF37_USE_CUDA -c -o $@ sf37_kvstore.c

rax.o: rax.c rax.h
	$(CC) $(CFLAGS) -DSF37_USE_CUDA -c -o $@ rax.c

sf37.o: sf37.c sf37.h sf37_ops.h sf37_quant.h sf37_cuda.h
	$(CC) $(CFLAGS) -DSF37_USE_CUDA -c -o $@ sf37.c

sf37_quant.o: sf37_quant.c sf37_quant.h
	$(CC) $(CFLAGS) -DSF37_USE_CUDA -c -o $@ sf37_quant.c

sf37_ops.o: sf37_ops.c sf37_ops.h sf37_quant.h
	$(CC) $(CFLAGS) -DSF37_USE_CUDA -c -o $@ sf37_ops.c

sf37_image.o: sf37_image.c sf37_image.h sf37.h third_party/stb_image.h
	$(CC) $(CFLAGS) -DSF37_USE_CUDA -c -o $@ sf37_image.c

sf37_cuda.o: sf37_cuda.cu sf37_cuda.h sf37_quant.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ sf37_cuda.cu

tests/test_sf37_quant: tests/test_sf37_quant.c sf37_quant.c sf37_quant.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_sf37_quant.c sf37_quant.c $(LDLIBS)

tests/test_sf37_ops: tests/test_sf37_ops.c sf37_ops.c sf37_ops.h sf37_quant.c sf37_quant.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_sf37_ops.c sf37_ops.c sf37_quant.c $(LDLIBS)

tests/test_sf37_payload.o: sf37.c sf37.h sf37_ops.h sf37_quant.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -DSF37_CPU_ONLY -DSF37_PAYLOAD_TEST -c -o $@ sf37.c

tests/test_sf37_payload: tests/test_sf37_payload.o sf37_quant_cpu.o sf37_ops_cpu.o sf37_image_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_sf37_server.o: sf37_server.c sf37.h sf37_image.h sf37_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -DSF37_CPU_ONLY -DSF37_SERVER_TEST -c -o $@ sf37_server.c

tests/test_sf37_server: tests/test_sf37_server.o sf37_kvstore_cpu.o rax_cpu.o sf37_cpu.o sf37_quant_cpu.o sf37_ops_cpu.o sf37_image_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test: tests/test_sf37_quant tests/test_sf37_ops tests/test_sf37_payload tests/test_sf37_server
	./tests/test_sf37_quant
	./tests/test_sf37_ops
	./tests/test_sf37_payload
	./tests/test_sf37_server

sf37_quant_cuda_test.o: sf37_quant.c sf37_quant.h
	$(CC) $(CFLAGS) -I. -DSF37_USE_CUDA -c -o $@ sf37_quant.c

sf37_ops_cuda_test.o: sf37_ops.c sf37_ops.h sf37_quant.h
	$(CC) $(CFLAGS) -I. -DSF37_USE_CUDA -c -o $@ sf37_ops.c

sf37_cuda_test.o: sf37_cuda.cu sf37_cuda.h sf37_quant.h
	$(NVCC) $(NVCCFLAGS) -I. -c -o $@ sf37_cuda.cu

tests/test_sf37_cuda_ops.o: tests/test_sf37_cuda_ops.cu sf37_cuda.h sf37_ops.h sf37_quant.h
	$(NVCC) $(NVCCFLAGS) -I. -c -o $@ tests/test_sf37_cuda_ops.cu

tests/test_sf37_cuda_ops: tests/test_sf37_cuda_ops.o sf37_cuda_test.o sf37_quant_cuda_test.o sf37_ops_cuda_test.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cuda-test: tests/test_sf37_cuda_ops
	LD_LIBRARY_PATH=$(CUDA_HOME)/lib:$(CUDA_HOME)/lib64:$$LD_LIBRARY_PATH ./tests/test_sf37_cuda_ops

inspect: cpu
	./sf37 --inspect --backend cpu --model ../quant/Step-3.7-Flash-MM-SF37-Q3GU-Q2D.gguf --tokenizer ../checkpoint/bf16

clean:
	rm -f sf37 sf37-bench sf37-server *.o tests/*.o tests/test_sf37_quant tests/test_sf37_ops tests/test_sf37_payload tests/test_sf37_server tests/test_sf37_cuda_ops
