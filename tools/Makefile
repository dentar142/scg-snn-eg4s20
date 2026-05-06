################################################################################
# Makefile - one-shot pipeline for SCG-CNN on Anlogic EG4S20
#
# Targets:
#   make data       Download CEBS, label and split into train/val
#   make train      Run INT8 QAT training
#   make export     Export weights as $readmemh files
#   make sim        Run iverilog simulation against test_window.hex
#   make synth      Run TD 6.2.2 batch build (synth + place + route + bitgen)
#   make load       Send weights+window to FPGA via UART (PORT=COMx required)
#   make all        data -> train -> export -> sim -> synth
#   make clean
################################################################################

PYTHON     ?= python
TDCMD      ?= tdcmd
PORT       ?= COM5

DATA_DIR   = data
CKPT_DIR   = model/ckpt
W_DIR      = rtl/weights
SIM_DIR    = sim
BUILD_DIR  = build

TRAIN_NPZ  = $(DATA_DIR)/train.npz
BEST_PT    = $(CKPT_DIR)/best.pt
WEIGHT_HEX = $(W_DIR)/L0_w.hex

.PHONY: all data train export sim synth load clean help

help:
	@echo "make {data | train | export | sim | synth | load | all | clean}"

# -- 1. dataset ----------------------------------------------------------------
$(TRAIN_NPZ):
	$(PYTHON) model/dataset_pipeline.py --out $(DATA_DIR)

data: $(TRAIN_NPZ)

# -- 2. QAT training -----------------------------------------------------------
$(BEST_PT): $(TRAIN_NPZ)
	$(PYTHON) model/train_qat.py --data $(DATA_DIR) --out $(CKPT_DIR)

train: $(BEST_PT)

# -- 3. weight export ----------------------------------------------------------
$(WEIGHT_HEX): $(BEST_PT)
	mkdir -p $(W_DIR)
	$(PYTHON) model/export_weights.py --ckpt $(BEST_PT) --data $(DATA_DIR) --out $(W_DIR)

export: $(WEIGHT_HEX)

# -- 4. simulation -------------------------------------------------------------
sim: export
	mkdir -p $(SIM_DIR)/work
	iverilog -g2012 -o $(SIM_DIR)/work/tb_scg_top.vvp \
		rtl/scg_top.v rtl/scg_mac_array.v sim/tb_scg_top.v
	cd $(SIM_DIR) && vvp work/tb_scg_top.vvp +TRACE=$(SIM_DIR)/work/wave.fst

# -- 5. synthesis (Anlogic TD 6.2.2) -------------------------------------------
synth: export
	mkdir -p $(BUILD_DIR)
	$(TDCMD) -batch tools/build.tcl

# -- 6. host-side test ---------------------------------------------------------
load:
	$(PYTHON) tools/test_inference.py --port $(PORT) \
		--weights $(W_DIR) \
		--window $(W_DIR)/test_window.hex

all: data train export sim synth

clean:
	rm -rf $(BUILD_DIR) $(SIM_DIR)/work $(W_DIR)/*.hex $(W_DIR)/*.mem
	@echo "kept $(DATA_DIR) and $(CKPT_DIR); 'make distclean' to wipe everything."

distclean: clean
	rm -rf $(DATA_DIR) $(CKPT_DIR)
