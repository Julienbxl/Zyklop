# =============================================================================
# Zyklop — BN254 Groth16 GPU Prover
# =============================================================================
#
# Targets:
#   make               — compile the prover (default)
#   make test          — run all unit tests
#   make bench         — run all benchmark circuits (Zyklop only)
#   make clean         — remove build directory
#   make help          — list all targets
#
# Build overrides:
#   make ARCH=sm_89    # RTX 4090
#   make ARCH=sm_86    # RTX 3090
#   make ARCH=sm_90    # H100
#   make ARCH=sm_120   # RTX 5060/5090 (default)
#
# Prerequisites:
#   CUDA toolkit >= 12.0, GMP (libgmp-dev), OpenMP (GCC)
#   circom 2.1+, snarkjs 0.7+, Node.js 18+ (for circuit setup only)

# =============================================================================
# Configuration
# =============================================================================

ARCH        ?= sm_120
NVCC        := nvcc
NVCCFLAGS   := -O3 -std=c++17 -arch=$(ARCH)

# Window size C : 17 is optimal on RTX 5060 (small-bucket dominant path)
# Override: make C=14 bench-poseidon-5000
C           ?= 17
NVCCFLAGS   += -DMSM_G1_C_BITS=$(C) -DC_BITS=$(C)

NVCCFLAGS   += -diag-suppress 177
INCLUDES    := -I include
LIBS        := -lgmp -Xcompiler -fopenmp

SRC_DIR     := src
INC_DIR     := include
TEST_DIR    := test
UNIT_DIR    := test/unit
BUILD_DIR   := build

PROVER_SRCS := $(SRC_DIR)/main.cu \
               $(SRC_DIR)/groth16.cu \
               $(SRC_DIR)/binfile_utils.cpp \
               $(SRC_DIR)/fileloader.cpp \
               $(SRC_DIR)/zkey_utils.cpp \
               $(SRC_DIR)/wtns_utils.cpp

# =============================================================================
# Build
# =============================================================================

.PHONY: all build test bench \
        bench-poseidon-5000 bench-sha256-50blocks bench-ecdsa4 bench-keccak40 \
        clean help

all: build

## build : compile the GPU prover
build: $(BUILD_DIR)/prover

$(BUILD_DIR)/prover: $(PROVER_SRCS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(PROVER_SRCS) $(LIBS) -o $@
	@echo "[OK] prover compiled -> $@"

# =============================================================================
# Unit tests
# =============================================================================

## test : run all unit tests (Fp, Fr, NTT, MSM G1, MSM G2)
test: test-fp test-fr test-ntt test-msm test-msm-g2
	@echo ""
	@echo "[OK] All unit tests passed."

TEST_BUILD := $(BUILD_DIR)/tests

test-fp: $(TEST_BUILD)/test_fp
	@echo "--- test_fp ---"
	$(TEST_BUILD)/test_fp $(UNIT_DIR)/test_vectors.json

$(TEST_BUILD)/test_fp: $(UNIT_DIR)/test_fp_bn254.cu
	mkdir -p $(TEST_BUILD)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

test-fr: $(TEST_BUILD)/test_fr
	@echo "--- test_fr ---"
	$(TEST_BUILD)/test_fr $(UNIT_DIR)/test_vectors_fr.json

$(TEST_BUILD)/test_fr: $(UNIT_DIR)/test_fr_bn254.cu
	mkdir -p $(TEST_BUILD)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

test-ntt: $(TEST_BUILD)/test_ntt
	@echo "--- test_ntt ---"
	$(TEST_BUILD)/test_ntt $(UNIT_DIR)/test_vectors_ntt.json

$(TEST_BUILD)/test_ntt: $(UNIT_DIR)/test_ntt.cu
	mkdir -p $(TEST_BUILD)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

test-msm: $(TEST_BUILD)/test_msm
	@echo "--- test_msm ---"
	$(TEST_BUILD)/test_msm $(UNIT_DIR)/test_vectors_msm.json

$(TEST_BUILD)/test_msm: $(UNIT_DIR)/test_msm.cu
	mkdir -p $(TEST_BUILD)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< $(LIBS) -o $@

test-msm-g2: $(TEST_BUILD)/test_msm_g2
	@echo "--- test_msm_g2 ---"
	$(TEST_BUILD)/test_msm_g2 $(UNIT_DIR)/test_vectors_msm_g2.json

$(TEST_BUILD)/test_msm_g2: $(UNIT_DIR)/test_msm_g2.cu
	mkdir -p $(TEST_BUILD)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< $(LIBS) -o $@

# =============================================================================
# Benchmarks
# =============================================================================
# Each target requires a pre-built circuit under test/<n>/.
# Run 'bash scripts/setup_benchmarks.sh <n>' first.
# Proof is verified with snarkjs after each run.

# RUN_IF_READY : run the bench only if the zkey exists, skip otherwise
define RUN_IF_READY
	@if [ -f test/$(1)/circuit_final.zkey ]; then \
		echo "--- Zyklop: $(3) ---"; \
		time $(BUILD_DIR)/prover test/$(1)/circuit_final.zkey \
		                         test/$(1)/witness.wtns \
		                         /tmp/proof.json /tmp/public.json && \
		snarkjs groth16 verify test/$(1)/verification_key.json \
		                       /tmp/public.json /tmp/proof.json; \
	else \
		echo "[skip] $(3) — run: bash scripts/setup_benchmarks.sh $(2)"; \
	fi
endef

## bench : run all four benchmark circuits
bench: build bench-poseidon-5000 bench-sha256-50blocks bench-ecdsa4 bench-keccak40

## bench-poseidon-5000 : Poseidon x5000 (~1.2M constraints)
bench-poseidon-5000: build
	$(call RUN_IF_READY,poseidon_5000,poseidon5000,Poseidon x5000)

## bench-sha256-50blocks : SHA-256 x50 (~3M constraints)
bench-sha256-50blocks: build
	$(call RUN_IF_READY,sha256_50blocks,50blocks,SHA-256 x50)

## bench-ecdsa4 : ECDSA verify x4 (~6M constraints)
bench-ecdsa4: build
	$(call RUN_IF_READY,ecdsa_4x,ecdsa4,ECDSA x4)

## bench-keccak40 : Keccak-256 x40 (~9.5M constraints)
bench-keccak40: build
	$(call RUN_IF_READY,keccak_40,keccak40,Keccak-256 x40)

# =============================================================================
# Clean
# =============================================================================

## clean : remove build directory
clean:
	rm -rf $(BUILD_DIR)
	@echo "[clean] Done."

# =============================================================================
# Help
# =============================================================================

## help : list available targets
help:
	@echo "Zyklop — BN254 Groth16 GPU Prover"
	@echo ""
	@echo "Usage: make [target] [ARCH=sm_XX] [C=17]"
	@echo ""
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "GPU architectures:"
	@echo "  sm_86  RTX 3090 / A10"
	@echo "  sm_89  RTX 4090 / L4"
	@echo "  sm_90  H100"
	@echo "  sm_120 RTX 5060/5090 (default)"
