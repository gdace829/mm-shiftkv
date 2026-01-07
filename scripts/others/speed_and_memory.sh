export BUDGET=64
export RATIO=0.1
export ENABLE_STATISTICAL_PREDICTOR=true
# export METHOD="sparsemm" fullkv
export METHOD="shiftkv"
CUDA_VISIBLE_DEVICES=0 python3 ./speed_and_memory.py
