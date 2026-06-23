#!/bin/bash
rm -rf ~/.triton/cache
rm -rf /root/.cache/vllm
export VLLM_ENGINE_READY_TIMEOUT_S=30000
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000
export IP_ADDRESS=$(hostname -I | awk '{print $1}')
export NETWORK_CARD_NAME="enp23s0f3"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_EXEC_TIMEOUT=60
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=$IP_ADDRESS
export GLOO_SOCKET_IFNAME=$NETWORK_CARD_NAME
export TP_SOCKET_IFNAME=$NETWORK_CARD_NAME
export HCCL_SOCKET_IFNAME=$NETWORK_CARD_NAME
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1500
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_TORCH_PROFILER_WITH_STACK=0
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_VERSION="0.18.0"

export VLLM_ASCEND_ENABLE_NZ=2
export ENABLE_QWEN3_5_MATMUL_NZ=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE=$((30 * 1024 * 1024))
export VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE=$((30 * 1024 * 1024))
export ENABLE_PREFILL_KV_STREAM=1
export MOE_COMM_TYPE="ALLGATHERREDUCESCATTER"
#export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1
#export ASCEND_GLOBAL_LOG_LEVEL=1
# export PYTHONPATH="/ce-efs-hd2-01/users/c00500727/va/qwen3_5/lib/vllm-ascend":$PYTHONPATH

export REUSE_PREFILLED_TOKENS=1

export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/turing_ascend_cloud/op_api/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}

# export VLLM_ASCEND_ENABLE_PHASE3=1
export VLLM_ASCEND_ENABLE_TRANSFER_MM_FEATURES=1
export ENABLE_SWIGLU_DYNAMIC_QUANT_OPS=1
export ENABLE_PREFILL_RESTORE_GDN_BF16=1

export VALIDATORS_CONFIG_PATH=/workspace/vllm-ascend/vllm_ascend/middleware/validator_config_vl.json

export VLLM_LOGGING_CONFIG_PATH=/home/ma-user/AscendCloud/logging_config.json

export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=1200

export ENABLE_LAYERNORM_DYNAMIC_QUANT_OPS=1

nohup vllm serve /ce-efs-hd2-01/users/l00956094/Qwen3.5-122B-A10B-linear-quant-w8a8-mtp \
      --host 0.0.0.0 \
      --port 8020 \
      --allowed-local-media-path / \
      --data-parallel-size 4 \
      --data-parallel-size-local 4 \
      --data-parallel-start-rank 0 \
      --api-server-count 1 \
      --data-parallel-address ${IP_ADDRESS} \
      --max-num-seqs 48 \
      --data-parallel-rpc-port 6984 \
      --tensor-parallel-size 2 \
      --seed 1024 \
      --middleware vllm_ascend.middleware.param_check.ValidateSamplingParams \
      --distributed-executor-backend mp \
      --served-model-name /ce-efs-hd2-01/users/l00956094/Qwen3.5-122B-A10B-linear-quant-w8a8-mtp \
      --max-model-len 128000 \
      --max-num-batched-tokens 16384 \
      --trust-remote-code \
      --quantization ascend \
      --no-disable-hybrid-kv-cache-manager \
      --async-scheduling \
      --no-enable-prefix-caching \
      --enable-expert-parallel \
      --reasoning-parser glm45 \
      --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": "3", "enforce_eager": true,"use_local_argmax_reduction": true}' \
      --additional-config '{"recompute_scheduler_enable": true, "enable_cpu_binding":true, "multistream_overlap_shared_expert": true}' \
      --compilation-config '{"cudagraph_capture_sizes":[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --gpu-memory-utilization 0.9 \
      --kv-transfer-config \
                  '{"kv_connector": "MooncakeLayerwiseConnector",
                  "kv_buffer_device":"npu",
                  "kv_role": "kv_consumer",
                  "kv_port": "24010",
                  "engine_id": "0",
                  "kv_connector_extra_config": {
                  "kv_transfer_params_zmq": true
                  }
                  }' \
      --profiler-config \
                  '{"profiler": "torch",
                  "torch_profiler_dir": "./vllm_profile_d",
                  "torch_profiler_with_stack": false}' \
      > start-d.log 2>&1 &

