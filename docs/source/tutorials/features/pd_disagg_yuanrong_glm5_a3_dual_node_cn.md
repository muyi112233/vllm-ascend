# 基于 Yuanrong 的 GLM-5 W4A8 A3 双机 PD 分离部署

## 概述

本指南提供在 2 台 Atlas 800I A3 / Atlas 800T A3 服务器上部署 GLM-5 W4A8 模型、使用 1P1D（1 个 Prefill 节点 + 1 个 Decode 节点）架构，并叠加 Yuanrong Datasystem 作为 KV Pool 后端的参考方案。

本文给出的是一套便于直接落地的 A3 双机对称配置：

- P 节点：DP2，TP8
- D 节点：DP2，TP8

每台 A3 服务器使用 16 个 NPU device，`launch_online_dp.py` 会在单机内拉起 2 个本地数据并行副本，每个副本占用 8 个 device。PD 分离所需的跨节点 KV 传输由 `MooncakeConnectorV1` 完成，外部 KV Pool 由 `AscendStoreConnector` 对接 Yuanrong 提供。

> **前置条件**：本文使用的 Docker 镜像版本为 `vllm-ascend:0.18.0rc1-a3`。如本地尚未下载，请先参考下文“使用 Docker 运行”中的命令拉取对应镜像。
>
> **当前版本补丁要求**：在当前版本下使用 Yuanrong 多级缓存前，需要先获取单独提供的以下三个 patch 文件。建议先将 patch 上传到容器内固定目录 `/workspace/yuanrong_patches/`，然后依次执行：
>
> ```bash
> mkdir -p /workspace/yuanrong_patches
> cd /vllm-workspace/vllm
> git am /workspace/yuanrong_patches/0001-Bugfix-Fix-negative-local_cache_hit-in-P-D-disaggreg.patch
>
> cd /vllm-workspace//vllm-ascend
> git am /workspace/yuanrong_patches/0001-Implement-yuanrong-backend.patch
> git am /workspace/yuanrong_patches/0001-BugFix-0.18.0-KV-Pool-Fix-KV-Pool-not-putting-kv-cac.patch
> ```
>
> 这三个 patch 文件需要单独提供，并上传到上述目录。其中，`0001-Bugfix-Fix-negative-local_cache_hit-in-P-D-disaggreg.patch` 需要打到 `/vllm-workspace/vllm` 仓库下，用于修复 `local_cache_hit` 指标出现负值的问题；`0001-Implement-yuanrong-backend.patch` 用于补充 Yuanrong backend 支持；`0001-BugFix-0.18.0-KV-Pool-Fix-KV-Pool-not-putting-kv-cac.patch` 用于修复 vLLM v0.18.0 在 speculative decoding 场景下 KV Pool 未正确执行 KV Cache put / finalize 的问题，并规避后续 vLLM metrics 统计相关报错。若环境中已包含这些 patch 的改动，可跳过此步骤。

## 环境准备

### 硬件要求

- **2 × Atlas 800I A3 或 Atlas 800T A3 服务器**
- 每台服务器配备 **8 张 NPU 卡，共 16 个 NPU device**
- 两个节点间已配置 RoCE 或灵衢网络，且网络互通

### 软件要求

采用模型配套的 Docker 镜像，软件版本与 Docker 镜像内置版本保持一致，确保 HDK、固件等软件在配套范围内。
此外：

- CANN 版本建议至少高于 `8.5.0`
- HDK 版本建议至少高于 `25.5.0`
- 若希望在 A3 上启用 `ASCEND_ENABLE_USE_FABRIC_MEM=1`，建议满足 `HDK >= 26.0.0` 且 `CANN >= 9.0.0`

### 模型权重

下载 GLM-5 W4A8 模型权重并放置到指定目录，例如 `/home/models/GLM-5-w4a8/`。

> **模型下载地址**：[魔搭社区](https://modelscope.cn/models/Eco-Tech/GLM-5-w4a8)

## 使用 Docker 运行

本教程使用的 Docker 镜像版本为 `vllm-ascend:0.18.0rc1-a3`。如本地尚未下载，可先执行：

```bash
docker pull quay.io/ascend/vllm-ascend:0.18.0rc1-a3
```

如果下载较慢，可将 `quay.io` 替换为 `m.daocloud.io/quay.io` 或 `quay.nju.edu.cn` 以加速拉取。更多镜像说明可参考[安装文档](../../installation.md#set-up-using-docker)。

在两个节点分别保存同一份 `start-docker.sh`：

```bash
#!/bin/bash
IMAGES_ID="$1"
NAME="$2"

if [ $# -ne 2 ]; then
    echo "error: 需要传入2个参数，格式：$0 <镜像ID> <容器名>"
    exit 1
fi

if ! docker images --format "{{.ID}}" | grep -q "^${IMAGES_ID:0:12}$"; then
    echo "error: 镜像ID $IMAGES_ID 不存在"
    exit 1
fi

docker run --name "${NAME}" -it -d --net=host --shm-size=800g \
    --privileged=true \
    -w /home \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp:/tmp \
    -v /data:/data \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -e http_proxy="$http_proxy" \
    -e https_proxy="$https_proxy" \
    "${IMAGES_ID}"
```

查看镜像 ID：

```bash
docker images | grep vllm-ascend
```

然后分别创建容器，例如：

```bash
# 节点 0（Prefill）
bash start-docker.sh <镜像ID> glm5-v0.18.0rc1-a3-p

# 节点 1（Decode）
bash start-docker.sh <镜像ID> glm5-v0.18.0rc1-a3-d
```

进入容器：

```bash
# 节点 0
docker exec -it glm5-v0.18.0rc1-a3-p bash

# 节点 1
docker exec -it glm5-v0.18.0rc1-a3-d bash
```

## 安装 Yuanrong Datasystem

```bash
wget https://gitcode.com/openeuler/yuanrong-datasystem/releases/download/v0.7.6.rc1/openyuanrong_datasystem-0.7.6rc1-cp311-cp311-manylinux_2_35_aarch64.whl
pip install openyuanrong_datasystem-0.7.6rc1-cp311-cp311-manylinux_2_35_aarch64.whl
```

验证安装：

```bash
python -c "import yr.datasystem; print('Yuanrong Datasystem 安装成功')"
```

## 安装 etcd

后续 Yuanrong 服务启动脚本依赖 `etcd` 和 `etcdctl`。A3 双机场景建议仅在 P 节点安装并启动 etcd，D 节点复用同一个 etcd 即可。

```bash
ETCD_VERSION="v3.5.12"
if [ "$(uname -m)" = "aarch64" ]; then
  ETCD_ARCH="linux-arm64"
else
  ETCD_ARCH="linux-amd64"
fi
wget https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
tar -xvf etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
cd etcd-${ETCD_VERSION}-${ETCD_ARCH}
cp etcd etcdctl /usr/local/bin/
```

验证安装：

```bash
etcd --version
etcdctl version
```

## 启动 Yuanrong 服务

### 节点 0（P 节点）

创建 `run_yr_node0.sh`，同时启动 etcd 和 Datasystem Worker：

```bash
#!/bin/bash

export HOST_IP="192.168.10.10"
export ETCD_IP="${HOST_IP}"
export WORKER_PORT=18481
export ETCD_PORT=2379
export SHM_SIZE=512000
export NODE_TIMEOUT=30
export NODE_DEAD_TIMEOUT=60
export LIVENESS_PATH=/workspace/liveness
mkdir -p ${LIVENESS_PATH}

etcd \
  --name etcd-single \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://${ETCD_IP}:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://${ETCD_IP}:2380 \
  --initial-cluster etcd-single=http://${ETCD_IP}:2380 \
  > /tmp/etcd.log 2>&1 &

sleep 3

etcdctl --endpoints "${ETCD_IP}:2379" put key "value"
etcdctl --endpoints "${ETCD_IP}:2379" get key

dscli start -w \
  --worker_address ${HOST_IP}:${WORKER_PORT} \
  --etcd_address ${ETCD_IP}:${ETCD_PORT} \
  --shared_memory_size_mb ${SHM_SIZE} \
  --node_timeout_s ${NODE_TIMEOUT} \
  --node_dead_timeout_s ${NODE_DEAD_TIMEOUT} \
  --liveness_check_path ${LIVENESS_PATH}
```

### 节点 1（D 节点）

创建 `run_yr_node1.sh`，只启动 Datasystem Worker：

```bash
#!/bin/bash

export HOST_IP="192.168.10.11"
export ETCD_IP="192.168.10.10"
export WORKER_PORT=18481
export ETCD_PORT=2379
export SHM_SIZE=512000
export NODE_TIMEOUT=30
export NODE_DEAD_TIMEOUT=60
export LIVENESS_PATH=/workspace/liveness
mkdir -p ${LIVENESS_PATH}

dscli start -w \
  --worker_address ${HOST_IP}:${WORKER_PORT} \
  --etcd_address ${ETCD_IP}:${ETCD_PORT} \
  --shared_memory_size_mb ${SHM_SIZE} \
  --node_timeout_s ${NODE_TIMEOUT} \
  --node_dead_timeout_s ${NODE_DEAD_TIMEOUT} \
  --liveness_check_path ${LIVENESS_PATH}
```

运行顺序：

```bash
# 节点 0
bash run_yr_node0.sh

# 节点 1
bash run_yr_node1.sh
```

## PD 分离部署（A3 双机、1P1D + Yuanrong）

### 并行策略

| 节点 | 角色 | NPU 资源 | 并行策略 | 本地服务端口 |
|------|------|----------|----------|--------------|
| 节点 0 | Prefill | 16 个 device | DP2 / TP8 | 6600、6601 |
| 节点 1 | Decode | 16 个 device | DP2 / TP8 | 6600、6601 |

### 准备脚本

建议在两个节点都创建同一工作目录，并从仓库中拷贝所需脚本：

```bash
mkdir -p /workspace/glm5-a3-pd-yuanrong
cd /workspace/glm5-a3-pd-yuanrong

cp /vllm-workspace/vllm-ascend/examples/external_online_dp/launch_online_dp.py .
cp /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py .
```

然后在该目录中按本文分别创建：

- `run_dp_template.sh`
- `server.sh`
- `proxy.sh`（仅节点 0 需要）

> `launch_online_dp.py` 会在当前目录查找 `./run_dp_template.sh`，因此请务必在同一目录下执行。

### kv_port 取值说明

A3 节点默认使用 16 个 NPU device。Mooncake 使用 AscendDirectTransport 进行 RDMA 数据传输时，会在 `[20000, 20000 + npu_per_node × 1000)` 范围内随机占用端口；对于 A3 16 device 场景，该保留区间即 `[20000, 35999]`。因此本文示例统一使用：

- P 节点 `kv_port=36000`
- D 节点 `kv_port=36200`

如果 `kv_port` 落在保留区间，启动阶段可能出现 `zmq.error.ZMQError: Address already in use`。

### 节点 0：Prefill

创建 `run_dp_template.sh`：

```bash
#!/bin/bash

# 网络配置 - 按实际环境修改
nic_name="enp67s0f0np0"
local_ip="192.168.10.10"
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

# NPU / vLLM 通用配置
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256

export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000

export ASCEND_RT_VISIBLE_DEVICES=$1

# Mooncake KV Transfer 配置
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export ASCEND_BUFFER_POOL=4:8

export PYTHONHASHSEED=0
# Yuanrong 配置
export DS_WORKER_ADDR="${local_ip}:18481"
export DS_H2D_MEMCPY_POLICY="direct"
export DS_D2H_MEMCPY_POLICY="direct"
unset GOOGLE_LOGTOSTDERR GOOGLE_ALSOLOGTOSTDERR

MODEL_PATH="/home/models/GLM-5-w4a8"

KV_TRANSFER_CONFIG=$(cat <<EOF
{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "engine_id": "$4",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "kv_port": "36000",
                "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                    },
                    "decode": {
                        "dp_size": 2,
                        "tp_size": 8
                    }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port": "$4",
                    "backend": "yuanrong"
                }
            }
        ]
    }
}
EOF
)

vllm serve $MODEL_PATH \
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
  --served-model-name glm-5 \
  --max-num-seqs 64 \
  --max-model-len 202752 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --quantization ascend \
  --async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --additional-config '{"enable_npugraph_ex": true, "fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable" : true}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  2>&1 | tee "./glm-5_yuanrong_p_dp${4}.log"
```

创建 `server.sh`：

```bash
#!/bin/bash
cd /workspace/glm5-a3-pd-yuanrong
python launch_online_dp.py \
  --dp-size 2 \
  --tp-size 8 \
  --dp-size-local 2 \
  --dp-rank-start 0 \
  --dp-address 192.168.10.10 \
  --dp-rpc-port 10521 \
  --vllm-start-port 6600
```

### 节点 1：Decode

创建 `run_dp_template.sh`：

```bash
#!/bin/bash

# 网络配置 - 按实际环境修改
nic_name="enp67s0f0np0"
local_ip="192.168.10.11"
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

# NPU / vLLM 通用配置
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256

export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000

export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export VLLM_ASCEND_ENABLE_MLAPO=1

# Mooncake KV Transfer 配置
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export PYTHONHASHSEED=0
# Yuanrong 配置
export DS_WORKER_ADDR="${local_ip}:18481"
export DS_H2D_MEMCPY_POLICY="direct"
export DS_D2H_MEMCPY_POLICY="direct"
unset GOOGLE_LOGTOSTDERR GOOGLE_ALSOLOGTOSTDERR

MODEL_PATH="/home/models/GLM-5-w4a8"

KV_TRANSFER_CONFIG=$(cat <<EOF
{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_consumer",
    "engine_id": "$4",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_consumer",
                "kv_port": "36200",
                "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                    },
                    "decode": {
                        "dp_size": 2,
                        "tp_size": 8
                    }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port": "$4",
                    "backend": "yuanrong"
                }
            }
        ]
    }
}
EOF
)

vllm serve $MODEL_PATH \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name glm-5 \
  --max-num-seqs 8 \
  --max-model-len 202752 \
  --max-num-batched-tokens 32 \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --quantization ascend \
  --async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4, 8, 12, 16,20,24,28, 32]}' \
  --additional-config '{"enable_npugraph_ex": true, "fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable" : true}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  2>&1 | tee "./glm-5_yuanrong_d_dp${4}.log"
```

创建 `server.sh`：

```bash
#!/bin/bash
cd /workspace/glm5-a3-pd-yuanrong
python launch_online_dp.py \
  --dp-size 2 \
  --tp-size 8 \
  --dp-size-local 2 \
  --dp-rank-start 0 \
  --dp-address 192.168.10.11 \
  --dp-rpc-port 10521 \
  --vllm-start-port 6600
```

### 节点 0：启动代理

创建 `proxy.sh`：

```bash
#!/bin/bash
cd /workspace/glm5-a3-pd-yuanrong
unset http_proxy
unset https_proxy

python load_balance_proxy_server_example.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts 192.168.10.10 192.168.10.10 \
  --prefiller-ports 6600 6601 \
  --decoder-hosts 192.168.10.11 192.168.10.11 \
  --decoder-ports 6600 6601
```

### 启动顺序

```bash
# 1. 节点 0：启动 etcd + Yuanrong Worker
bash run_yr_node0.sh

# 2. 节点 1：启动 Yuanrong Worker
bash run_yr_node1.sh

# 3. 节点 0：启动 Prefill 实例
bash server.sh

# 4. 节点 1：启动 Decode 实例
bash server.sh

# 5. 节点 0：启动代理
bash proxy.sh > proxy.log 2>&1 &
```

## 关键参数说明

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `dp-size` | `2` | P 组和 D 组各自的全局数据并行大小 |
| `dp-size-local` | `2` | 每个节点拉起 2 个本地 DP 副本 |
| `tensor-parallel-size` | `8` | 每个 DP 副本使用 8 个 NPU device |
| `kv_port` | P=`36000`，D=`36200` | A3 16 device 场景建议避开 `[20000, 35999]` |
| `lookup_rpc_port` | `$4` | 直接复用当前 DP rank，保证同机唯一 |
| `engine_id` | `$4` | 建议与当前 DP rank 保持一致，便于排障 |
| `ASCEND_ENABLE_USE_FABRIC_MEM` | `1` | A3 推荐配置，开启统一地址直传 |
| `PYTHONHASHSEED` | `0` | 两个节点必须保持一致，确保 KV Cache 键一致性 |

### MultiConnector 配置要点

本文使用 `MultiConnector` 组合两类能力：

1. `MooncakeConnectorV1`：负责 P/D 节点间的 KV Transfer。
2. `AscendStoreConnector`：负责接入 Yuanrong 外部 KV Pool。

使用时需要注意：

- 顶层 `kv_role` 与子连接器 `kv_role` 必须一致。
- 同一台机器上不同 DP 副本的 `lookup_rpc_port` 必须唯一。
- P/D 两侧的 `prefill.dp_size/tp_size` 和 `decode.dp_size/tp_size` 要保持一致。
- A3 场景不要继续沿用 A2 文档中的 `30000/30100` 端口组合。

## 功能验证

### 检查代理健康状态

```bash
curl http://192.168.10.10:8000/healthcheck
```

### 测试推理

```bash
curl -H "Accept: application/json" \
    -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "glm-5",
        "messages": [{
            "role": "user",
            "content": "你好，请介绍一下人工智能在软件工程中的典型应用。"
        }],
        "stream": false,
        "ignore_eos": false,
        "temperature": 0,
        "max_tokens": 200
    }' http://192.168.10.10:8000/v1/chat/completions
```

## 缓存命中率监控

### 查看后端日志

推荐直接查看后端实例日志，而不是代理日志。本文示例日志文件名如下：

- P 节点：`glm-5_yuanrong_p_dp0.log`、`glm-5_yuanrong_p_dp1.log`
- D 节点：`glm-5_yuanrong_d_dp0.log`、`glm-5_yuanrong_d_dp1.log`

```bash
tail -f glm-5_yuanrong_d_dp0.log
tail -f glm-5_yuanrong_d_dp0.log | grep -E "Prefix cache hit rate|External prefix cache hit rate|num_computed_tokens"
```

### 查看后端 metrics

代理端不聚合 vLLM metrics，建议直接查看后端实例：

```bash
curl http://192.168.10.11:6600/metrics | grep external_prefix_cache
```

如果当前环境包含 `vllm-ascend` 仓库源码，也可直接使用仓库脚本：

```bash
bash tools/watch_cache_hit_rate.sh -u http://192.168.10.11:6600/metrics -i 10
```

## 故障排除

### 常见问题

1. **启动时报 `zmq.error.ZMQError: Address already in use`**

   优先检查 `kv_port` 是否仍落在 A3 保留区间 `[20000, 35999]` 内。建议使用本文示例中的 `36000/36200`。

2. **etcd 连接失败**

   在节点 0 检查 etcd 是否健康：
   ```bash
   etcdctl --endpoints "192.168.10.10:2379" endpoint health
   ```

3. **Worker 注册失败**

   检查两个节点的 Worker 是否已监听：
   ```bash
   netstat -tlnp | grep 18481
   ```

4. **同机两个实例的 Yuanrong 查询端口冲突**

   确认 `lookup_rpc_port` 没有写死成同一个值。本文示例直接使用 `$4`，即当前 DP rank。

5. **KV Cache 未命中或命中率异常低**

   确认两个节点都设置了相同的哈希种子：
   ```bash
   echo $PYTHONHASHSEED
   ```

6. **引擎启动超时**

   可以先放宽等待时间：
   ```bash
   export VLLM_ENGINE_READY_TIMEOUT_S=3600
   ```

7. **节点时间不一致**

   双机场景下建议两个节点的系统时间保持一致：
   ```bash
   date
   timedatectl
   ```

## 参考资料

- [基于 Yuanrong 的 GLM-5 W4A8 单实例部署](pd_colocated_yuanrong_glm5_cn.md)
- [基于 Yuanrong 的 GLM-5 W8A8 8机 A2 大EP PD分离部署](pd_disagg_yuanrong_glm5_cn.md)
- [external_online_dp README](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/README.md)
- [Yuanrong Datasystem 文档](https://atomgit.com/openeuler/yuanrong-datasystem)
- [etcd 文档](https://etcd.io/docs/)
- [vLLM Ascend 文档](https://docs.vllm.ai/projects/ascend/)
