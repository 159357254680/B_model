#!/bin/bash
set -e

MODE=${1:-full}

case $MODE in
  mock)
    echo "=== Docker: mock 模式 ==="
    docker compose run --rm b_model python train.py --mock
    ;;
  full)
    echo "=== Docker: 完整流程（A 预处理 + B 训练 + 分析）==="
    docker compose run --rm b_model bash -c "
      python a_preprocess.py && python train.py
    "
    ;;
  interactive)
    echo "=== Docker: 交互模式 ==="
    docker compose run --rm b_model bash
    ;;
  *)
    echo "用法: ./run_docker.sh [mock|full|interactive]"
    echo "  mock        - 模拟数据快速测试（不需要下载 IMDB）"
    echo "  full        - 下载 IMDB 真实数据 + 完整训练分析"
    echo "  interactive - 进入容器 bash 手动操作"
    exit 1
    ;;
esac
