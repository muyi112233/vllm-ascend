# GLM-5 W8A8 A2 部署调优实践

**关键词**：GLM-5、vLLM-Ascend、Atlas 800 A2、性能调优

**作者**：潘嘉琪(00512755)、石少锋(00645958)、蔡琪刚(00625726)、汪莉晴(00643945)、刘晨冰(00848480)、张良壮(00500064)

## 1. 背景描述

### 1.1 模型描述

GLM-5 是智谱新一代的旗舰基座模型，面向 Agentic Engineering 打造，能够在复杂系统工程与长程 Agent 任务中提供可靠生产力。在 Coding 与 Agent 能力上，GLM-5 取得开源 SOTA 表现，在真实编程场景的使用体感逼近 Claude Opus 4.5，擅长复杂系统工程与长程 Agent 任务，是通用 Agent 助手的理想基座。

### 1.2 环境信息

| 组件 | 版本 | 备注 |
|------|------|------|
| 服务器硬件 | Atlas 800 A2 | 8机PD分离、4机混部 |
| vLLM-Ascend | quay.io/ascend/vllm-ascend:main 合入 PR 7139 | |
| HDK | 25.2.0 | |
| CANN | 8.2.RC1 | |
| GLM-5权重 | W8A8量化 | [ModelScope](https://modelscope.cn/models/umiiiiii/GLM-W8A8/files) |

## 2. 叠加特性优化

| 优化特性 | 使能方法 |
|----------|----------|
| W8A8模型量化 | [ModelScope权重](https://modelscope.cn/models/umiiiiii/GLM-W8A8/files) |
| FLASHCOMM1算子接入 | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` |
| 异步调度 | vllm启动命令中 `--async-scheduling` |
| MLAPO算子接入 | [PR #6902](https://github.com/vllm-project/vllm-ascend/pull/6902) 适配接入 |
| mul_add融合算子使能 | [PR #5518](https://github.com/vllm-project/vllm-ascend/pull/5518) / [PR #6928](https://github.com/vllm-project/vllm-ascend/pull/6928) 适配接入，`--additional-config` 中加 `"fuse_muls_add": true` |
| PD分离 | 1P1D：P: DP4/TP8，D: DP8/TP4 |
| 共享专家多流 | `--additional-config` 中加 `"recompute_scheduler_enable": true` + `"multistream_overlap_shared_expert": true` |
| MTP接受率提升 | `"multistream_overlap_shared_expert": true` + [PR](https://github.com/yydyzr/vllm/pull/2)，`--additional-config` 中加 `"rot_path": "xxx/rot.safetensors"`，`--speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}'` |
| MTP-DP入图 | `"fuse_qknorm_rope": false`，[PR #6948](https://github.com/vllm-project/vllm-ascend/pull/6948) |
| 流水优化 | `"fuse_muls_add": true` + `export TASK_QUEUE_ENABLE=1` |
| 通信算法AIV | `--additional-config` 中加 `"enable_npugraph_ex": true` + `export HCCL_OP_EXPANSION_MODE="AIV"` |
| FULL_DECODE_ONLY（仅D节点） | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]}'` |

## 3. 部署指导

### 3.1 创建容器

创建 `docker.sh` 启动脚本：

```bash
export IMAGE=a480062695ec

docker run --privileged \
    --name glm-5 \
    --shm-size=500g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /mnt/:/mnt/ \
    -v /opt/:/opt/ \
    -v /home:/home \
    -itd $IMAGE bash

# 运行容器
bash docker.sh

# 进入容器
docker exec -it glm-5 bash
```

### 3.2 PD分离部署（8机、1P1D）

**并行策略**：
- P实例：DP4，TP8
- D实例：DP8，TP4

**节点分配**：

| 节点 | IP | 需要文件 |
|------|-----|----------|
| P节点（主） | 71.10.29.138 | launch_online_dp.py、run_dp_template.sh、server.sh、proxy.sh、load_balance_proxy_server_example.py |
| P节点（从） | 71.10.29.141 | launch_online_dp.py、run_dp_template.sh、server.sh |
| P节点（从） | 71.10.29.125 | launch_online_dp.py、run_dp_template.sh、server.sh |
| P节点（从） | 71.10.29.128 | launch_online_dp.py、run_dp_template.sh、server.sh |
| D节点（主） | 71.10.29.124 | launch_online_dp.py、run_dp_template.sh、server.sh |
| D节点（从） | 71.10.29.123 | launch_online_dp.py、run_dp_template.sh、server.sh |
| D节点（从） | 71.10.29.139 | launch_online_dp.py、run_dp_template.sh、server.sh |
| D节点（从） | 71.10.29.142 | launch_online_dp.py、run_dp_template.sh、server.sh |

**脚本说明**：
- `launch_online_dp.py`：每个节点都要有，无需修改
- `run_dp_template.sh`：每个节点根据实际情况修改
- `dp_load_balance_proxy_server.py`：仅P主节点需要

详细说明见：[external_online_dp README](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/README.md)

#### 3.2.1 P节点

`run_dp_template.sh` 模板，请按实际情况修改 `nic_name`、`local_ip`、权重路径、`rot_path`、`torch_profiler_dir`：

```bash
rm -rf ~/ascend
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export PYTHONPATH=/workspace/vllm-ascend_3_5/vllm-ascend:${PYTHONPATH}
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"

nic_name="enp67s0f0np0"
local_ip=71.10.29.138
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

# Mooncake
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export ASCEND_CONNECT_TIMEOUT=300000
export ASCEND_TRANSFER_TIMEOUT=300000
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

# optim
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve /opt/data/verification/models/GLM-W8A8 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --enable-chunked-prefill \
    --seed 1024 \
    --served-model-name glm5 \
    --max-model-len 133120 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --async-scheduling \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "engine_id": "0",
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
        "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {
                "dp_size": 4,
                "tp_size": 8
            },
            "decode": {
                "dp_size": 8,
                "tp_size": 4
            }
        }
    }' \
    --additional-config \
    '{
        "recompute_scheduler_enable": true,
        "multistream_overlap_shared_expert": true,
        "fuse_qknorm_rope": false,
        "fuse_muls_add": true,
        "enable_npugraph_ex": true,
        "rot_path": "/home/glm5/scripts/rot.safetensors"
    }' \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
    --profiler-config \
    '{
        "profiler": "torch",
        "torch_profiler_dir": "/home/glm5/profiling",
        "torch_profiler_with_stack": false
    }' \
    2>&1 | tee glm.log
```

**server.sh**：P实例 DP4、TP8

```bash
# 71.10.29.138 P 主节点
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address 71.10.29.138 --dp-rpc-port 10521 --vllm-start-port 6600

# 71.10.29.141 P 从节点
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address 71.10.29.138 --dp-rpc-port 10521 --vllm-start-port 6600

# 71.10.29.125 P 从节点
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 2 --dp-address 71.10.29.138 --dp-rpc-port 10521 --vllm-start-port 6600

# 71.10.29.128 P 从节点
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 3 --dp-address 71.10.29.138 --dp-rpc-port 10521 --vllm-start-port 6600
```

**proxy.sh**：只存在于P主节点，在P/D实例服务启动成功后执行 `bash proxy.sh > proxy.log &`，根据实际情况修改组网IP。

```bash
unset http_proxy
unset https_proxy
python load_balance_proxy_server_example.py \
    --port 8000 \
    --host 0.0.0.0 \
    --prefiller-hosts \
        71.10.29.138 \
        71.10.29.141 \
        71.10.29.128 \
        71.10.29.125 \
    --prefiller-ports \
        6600 \
        6600 \
        6600 \
        6600 \
    --decoder-hosts \
        71.10.29.124 \
        71.10.29.124 \
        71.10.29.142 \
        71.10.29.142 \
        71.10.29.139 \
        71.10.29.139 \
        71.10.29.123 \
        71.10.29.123 \
    --decoder-ports \
        6600 6601 \
        6600 6601 \
        6600 6601 \
        6600 6601
```

#### 3.2.2 D节点

`run_dp_template.sh` 模板，请按实际情况修改 `nic_name`、`local_ip`、权重路径、`rot_path`、`torch_profiler_dir`：

```bash
rm -rf ~/ascend
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export PYTHONPATH=/workspace/vllm-ascend_3_5/vllm-ascend:${PYTHONPATH}
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"

nic_name="enp67s0f0np0"
local_ip=71.10.29.124
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

# Mooncake
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export ASCEND_CONNECT_TIMEOUT=300000
export ASCEND_TRANSFER_TIMEOUT=300000
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

# optim
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve /opt/data/verification/models/GLM-W8A8 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm5 \
    --max-model-len 133120 \
    --max-num-batched-tokens 32 \
    --trust-remote-code \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    --quantization ascend \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": "30100",
        "engine_id": "1",
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
        "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {
                "dp_size": 4,
                "tp_size": 8
            },
            "decode": {
                "dp_size": 8,
                "tp_size": 4
            }
        }
    }' \
    --compilation-config \
    '{
        "cudagraph_capture_sizes": [1,4,8,12,16,20,24,28,32,36,40,48,56,64,80,96],
        "cudagraph_mode": "FULL_DECODE_ONLY"
    }' \
    --additional-config \
    '{
        "recompute_scheduler_enable": true,
        "multistream_overlap_shared_expert": true,
        "fuse_qknorm_rope": false,
        "fuse_muls_add": true,
        "enable_npugraph_ex": true,
        "rot_path": "/home/glm5/scripts/rot.safetensors"
    }' \
    --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
    --profiler-config \
    '{
        "profiler": "torch",
        "torch_profiler_dir": "/home/glm5/profiling",
        "torch_profiler_with_stack": false
    }' \
    2>&1 | tee glm.log
```

**server.sh**：D实例 DP8、TP4

```bash
# 71.10.29.124 D 主节点
python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 0 --dp-address 71.10.29.124 --dp-rpc-port 10521 --vllm-start-port 6600

# 71.10.29.123 D 从节点
python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 2 --dp-address 71.10.29.124 --dp-rpc-port 10521 --vllm-start-port 6600

# 71.10.29.139 D 从节点
python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 4 --dp-address 71.10.29.124 --dp-rpc-port 10521 --vllm-start-port 6600

# 71.10.29.142 D 从节点
python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 6 --dp-address 71.10.29.124 --dp-rpc-port 10521 --vllm-start-port 6600
```

## 4. 性能基线

### 4.1 P-DP4-TP8 / D-DP8-TP4 / Prefix cache 0% / MTP3（无优化）

| 输入 | 输出 | 并发 | Mean TTFT (ms) | Median TTFT (ms) | P90 TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P90 TPOT (ms) | P99 TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | P90 ITL (ms) | P99 ITL (ms) |
|------|------|------|----------------|------------------|---------------|---------------|----------------|------------------|---------------|---------------|---------------|-----------------|--------------|--------------|
| 3.5K | 1.5K | 10 | 2249.30 | 1948.79 | 3704.12 | 4171.17 | 49.28 | 46.39 | 64.62 | 79.00 | 79.31 | 75.47 | 99.99 | 193.04 |
| 32K | 1K | 10 | 13858.68 | 11791.30 | 19927.97 | 27614.58 | 76.04 | 76.98 | 79.00 | 80.33 | 80.07 | 77.24 | 106.66 | 194.77 |
| 32K | 2K | 10 | 13381.57 | 11622.34 | 19549.19 | 26983.33 | 75.71 | 76.09 | 77.58 | 77.84 | 79.14 | 77.18 | 103.95 | 173.72 |
| 64K | 1K | 10 | 26439.01 | 22437.97 | 38042.62 | 55339.73 | 79.57 | 80.74 | 83.13 | 85.40 | 82.89 | 80.62 | 105.44 | 184.02 |
| 64K | 2K | 10 | 25695.28 | 22302.24 | 39395.78 | 56534.29 | 77.60 | 79.96 | 81.40 | 82.17 | 82.10 | 80.56 | 108.12 | 168.63 |

### 4.2 P-DP4-TP8 / D-DP8-TP4 / Prefix cache 90% / MTP3（无优化）

| 输入 | 输出 | 并发 | Mean TTFT (ms) | Median TTFT (ms) | P90 TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P90 TPOT (ms) | P99 TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | P90 ITL (ms) | P99 ITL (ms) |
|------|------|------|----------------|------------------|---------------|---------------|----------------|------------------|---------------|---------------|---------------|-----------------|--------------|--------------|
| 3.5K | 1.5K | 10 | 2206.77 | 1941.75 | 3333.27 | 4469.28 | 49.99 | 47.43 | 62.69 | 77.29 | 79.15 | 75.37 | 100.85 | 195.94 |
| 32K | 1K | 10 | 2829.91 | 2296.63 | 4814.02 | 5387.56 | 55.92 | 54.84 | 66.12 | 85.09 | 84.45 | 79.02 | 107.03 | 493.89 |
| 32K | 2K | 10 | 3993.39 | 2253.39 | 11545.60 | 17175.45 | 53.26 | 52.29 | 73.42 | 86.09 | 84.75 | 79.35 | 108.49 | 204.45 |
| 64K | 1K | 10 | 4585.26 | 3990.01 | 7290.68 | 9458.98 | 57.04 | 56.69 | 71.08 | 87.38 | 88.63 | 83.12 | 110.19 | 532.27 |
| 64K | 2K | 10 | 5151.09 | 3893.12 | 12075.55 | 13530.88 | 55.02 | 54.60 | 69.26 | 77.27 | 85.50 | 81.61 | 110.05 | 197.75 |
| 128K | 1K | 10 | 432291.03 | 396537.29 | 669657.12 | 754730.19 | 66.60 | 67.95 | 77.09 | 82.33 | 79.87 | 76.96 | 108.68 | 180.88 |
| 128K | 2K | 10 | 529063.87 | 543963.16 | 756999.55 | 908553.75 | 68.33 | 68.52 | 84.34 | 117.99 | 89.84 | 77.39 | 107.99 | 232.54 |

### 4.3 P-DP4-TP8 / D-DP8-TP4 / Prefix cache 0% / MTP1（优化后）

| 输入 | 输出 | 并发 | Mean TTFT (ms) | Median TTFT (ms) | P90 TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P90 TPOT (ms) | P99 TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | P90 ITL (ms) | P99 ITL (ms) |
|------|------|------|----------------|------------------|---------------|---------------|----------------|------------------|---------------|---------------|---------------|-----------------|--------------|--------------|
| 2.05K | 0.87K | 20 | 2097.35 | 1647.73 | 3802.71 | 4478.28 | 48.87 | 47.49 | 56.34 | 71.38 | 72.39 | 72.31 | 75.70 | 120.10 |
| 2.05K | 0.87K | 23 | 2117.41 | 1624.71 | 3867.41 | 4623.33 | 48.80 | 45.53 | 63.88 | 73.04 | 73.24 | 73.15 | 77.51 | 130.41 |
| 3.5K | 1.5K | 10 | 1844.33 | 1562.59 | 3026.20 | 3784.20 | 49.25 | 43.92 | 72.15 | 73.73 | 73.63 | 73.52 | 76.38 | 136.27 |
| 3.5K | 1.5K | 11 | 1871.19 | 1537.71 | 3000.80 | 3801.20 | 50.28 | 45.81 | 71.84 | 72.16 | 73.38 | 73.43 | 75.64 | 135.63 |
| 20K | 2K | 1 | 4980.33 | 4936.05 | 5057.81 | 5085.21 | 72.99 | 72.95 | 73.94 | 74.16 | 76.01 | 74.97 | 108.14 | 154.98 |

**结论**（Prefix cache 0%，MTP=1）：
- 输入 2.1K，输出 0.9K，TPOT 卡 50ms：并发上限在 **23**
- 输入 3.5K，输出 1.5K，TPOT 卡 50ms：并发上限在 **10**
- 输入 20K，输出 2K，TPOT 卡 50ms：并发 1，TPOT 超 50ms，MTP=1 和 MTP=3 均不满足

### 4.4 P-DP4-TP8 / D-DP8-TP4 / Prefix cache 90% / MTP1（优化后）

| 输入 | 输出 | 并发 | Mean TTFT (ms) | Median TTFT (ms) | P90 TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P90 TPOT (ms) | P99 TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | P90 ITL (ms) | P99 ITL (ms) |
|------|------|------|----------------|------------------|---------------|---------------|----------------|------------------|---------------|---------------|---------------|-----------------|--------------|--------------|
| 2.05K | 0.87K | 35 | 1812.11 | 1871.66 | 2596.90 | 2714.30 | 47.64 | 46.65 | 53.12 | 67.53 | 74.67 | 75.09 | 78.78 | 119.46 |
| 2.05K | 0.87K | 40 | 1794.86 | 1821.93 | 2356.59 | 2667.53 | 47.35 | 46.56 | 53.01 | 69.84 | 75.63 | 75.75 | 80.89 | 129.88 |
| 2.05K | 0.87K | 45 | 1990.75 | 2040.87 | 2894.37 | 3634.47 | 52.20 | 49.88 | 65.85 | 76.98 | 79.02 | 79.30 | 84.11 | 134.26 |
| 3.5K | 1.5K | 40 | 1811.01 | 1719.17 | 2844.60 | 2935.61 | 48.22 | 47.11 | 55.24 | 74.02 | 76.66 | 77.04 | 81.99 | 131.65 |
| 3.5K | 1.5K | 45 | 1875.39 | 1898.29 | 2431.32 | 2800.85 | 48.93 | 47.78 | 56.51 | 75.57 | 78.47 | 79.00 | 83.50 | 128.00 |
| 20K | 2K | 40 | 2403.09 | 1844.59 | 4355.55 | 5953.90 | 48.66 | 47.08 | 58.41 | 73.32 | 78.70 | 79.13 | 83.05 | 123.20 |
| 20K | 2K | 45 | 2525.54 | 1659.52 | 5097.89 | 7366.47 | 50.16 | 48.82 | 58.86 | 76.94 | 81.67 | 82.19 | 86.55 | 130.78 |
| 20K | 2K | 50 | 2469.29 | 1737.67 | 5214.61 | 6473.43 | 53.04 | 50.53 | 66.22 | 79.89 | 83.48 | 83.89 | 89.30 | 141.15 |

**结论**（Prefix cache 90%，MTP=1）：
- 输入 2.1K，输出 0.9K，TPOT 卡 50ms：并发上限在 **40**
- 输入 3.5K，输出 1.5K，TPOT 卡 50ms：并发上限在 **45**
- 输入 20K，输出 2K，TPOT 卡 50ms：并发上限在 **45**

## 5. 常见问题

### 5.1 transformers 版本过低

**解决方案**：升级 transformer 版本

```bash
pip install transformers==5.2.0 --no-deps --force-reinstall
pip install huggingface_hub==1.5.0 --no-deps --force-reinstall
```

### 5.2 加了 rot.safetensors，报错 `KeyError: 'rot'`

**解决方案**：rot 的权重要放在其他目录，不能和 W8A8 的放一起。

### 5.3 部署上下文 200K，报超出模型最大支持长度

**问题现象**：

```
Value error, User-specified max_model_len (204800) is greater than the derived max_model_len
(max_position_embeddings = 202752.0 or model_max_length = None in model's config.json). To allow
overriding this maximum, set the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.

VLLM_ALLOW_LONG_MAX_MODEL_LEN must be used with extreme caution. If the model uses relative
position encoding (RoPE), positions exceeding derived_max_model_len lead to nan. If the
model uses absolute position encoding, positions exceeding derived_max_model_len will cause
a CUDA array out-of-bounds error.
```

**解决方案**：将 `--max-model-len` 参数调低，官方宣称最大支持 200K 上下文，实测最大只能到 **198K**。

### 5.4 PP并行策略报错

**问题现象**：

```
NotImplementedError: Pipeline parallelism is not supported for this model. Supported models
implement the `SupportsPP` interface.
```

**解决方案**：模型不支持 PP 并行策略，改用 **DP+TP** 并行策略。
