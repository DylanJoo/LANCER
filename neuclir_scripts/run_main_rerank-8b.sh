#!/bin/sh
#SBATCH --job-name=main-8b-qwen2-5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=%x.out

module load anaconda3/2024.2
conda activate ecir2026

# MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL=Qwen/Qwen2.5-7B-Instruct

# Start experiments
ss_arg_list=(
    "bm25@localhost:5000"
    "plaidx-neuclir@10.162.95.158:5000"
    # "qwen3-neuclir@10.162.95.158:5000"
    # "lsr-neuclir-mt@10.162.95.158:5000"
    "lsr-neuclir@10.162.95.158:5000"
)

for ss_args in "${ss_arg_list[@]}"; do
    service_name=$(echo $ss_args | cut -d'@' -f1)
    service_endpoint=$(echo $ss_args | cut -d'@' -f2)

    ## CRUXreranking
    NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 vllm serve $MODEL \
        --max-model-len 8196  \
        --port 8000  \
        --dtype float16 \
        --disable-custom-all-reduce \
        --tensor-parallel-size 1 >> vllm_server.log 2>&1 &
    PID=$!

    echo "Waiting for vLLM server (PID=$PID) to start..."
    until curl -s http://localhost:8000/v1/models >/dev/null; do
      sleep 10
    done

    python3 run_crux_reranker_eval.py \
        --host http://${service_endpoint} \
        --llm_base_url http://localhost:8000/v1 \
        --temperature 0.0 --top_p 1.0 \
        --model $MODEL \
        --reranker crux_reranking \
        --service_name ${service_name} \
        --dataset_name neuclir

    ## also the oracle
    python3 run_crux_reranker_eval.py \
        --host http://${service_endpoint} \
        --llm_base_url http://localhost:8000/v1 \
        --temperature 0.0 --top_p 1.0 \
        --model $MODEL \
        --reranker crux_reranking:oracle \
        --service_name ${service_name} \
        --dataset_name neuclir \
        --tag human

    kill $PID

    ## pointwise
    NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 vllm serve $MODEL \
        --max-model-len 8196  \
        --port 8000  \
        --dtype float16 \
        --disable-custom-all-reduce \
        --tensor-parallel-size 1 >> vllm_server.log 2>&1 &
    PID=$!

    echo "Waiting for vLLM server (PID=$PID) to start..."
    until curl -s http://localhost:8000/v1/models >/dev/null; do
      sleep 10
    done

    python3 run_crux_reranker_eval.py \
        --host http://${service_endpoint} \
        --llm_base_url http://localhost:8000/v1 \
        --temperature 0.0 --top_p 1.0 \
        --model $MODEL \
        --reranker autorerank:point:$MODEL \
        --service_name ${service_name} \
        --dataset_name neuclir
    kill $PID

    # setwise
    NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 vllm serve $MODEL \
        --max-model-len 10240  \
        --port 8000  \
        --dtype float16 \
        --disable-custom-all-reduce \
        --tensor-parallel-size 1 >> vllm_server.log 2>&1 &
    PID=$!

    echo "Waiting for vLLM server (PID=$PID) to start..."
    until curl -s http://localhost:8000/v1/models >/dev/null; do
      sleep 10
    done

    python3 run_crux_reranker_eval.py \
        --host http://${service_endpoint} \
        --llm_base_url http://localhost:8000/v1 \
        --temperature 0.0 --top_p 1.0 \
        --model $MODEL \
        --reranker autorerank:setmaxheaptopk:$MODEL \
        --service_name ${service_name} \
        --dataset_name neuclir
    kill $PID

    # listwise
    NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 vllm serve $MODEL \
        --max-model-len 20480  \
        --port 8000  \
        --dtype float16 \
        --disable-custom-all-reduce \
        --tensor-parallel-size 1 >> vllm_server.log 2>&1 &
    PID=$!

    echo "Waiting for vLLM server (PID=$PID) to start..."
    until curl -s http://localhost:8000/v1/models >/dev/null; do
      sleep 10
    done

    python3 run_crux_reranker_eval.py \
        --host http://${service_endpoint} \
        --llm_base_url http://localhost:8000/v1 \
        --temperature 0.0 --top_p 1.0 \
        --model $MODEL \
        --reranker autorerank:rankgpt:$MODEL \
        --service_name ${service_name} \
        --dataset_name neuclir
    kill $PID
done
