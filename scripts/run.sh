#!/bin/bash

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 激活conda环境
# conda activate transformer

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# 运行训练脚本
echo "Starting Transformer training..."
python "$PROJECT_ROOT/src/train.py" \
    --config "$PROJECT_ROOT/configs/base.yaml" \
    --seed 42

echo "Training completed!"