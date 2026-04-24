# 基于 Yuanrong 的 GLM-5.1 W8A8 A3 4机大EP PD分离部署

## 概述

本文提供在 4 台 Atlas 800I A3 / Atlas 800T A3 服务器上部署 GLM-5.1 W8A8 模型、使用 PD 分离架构，并叠加 Yuanrong Datasystem 作为 KV Pool 后端的参考方案。该方案使用 2 台 A3 作为 Prefill 组，2 台 A3 作为 Decode 组。

本文给出的并行策略如下：

| 分组 | 节点数 | 每节点 NPU device | 全局并行策略 | 每节点本地实例数 |
|------|--------|-------------------|--------------|------------------|
| Prefill | 2 | 16 | DP4 / TP8 / EP | 2 |
| Decode | 2 | 16 | DP8 / TP4 / EP | 4 |

其中 `--enable-expert-parallel` 用于启用 MoE 专家并行。Decode 侧使用更小的 TP 切分和更多 DP 副本，适合提升 Decode 阶段吞吐；Prefill 侧保持 TP8，适合长上下文预填充。PD 分离架构下，Prefill 节点与 Decode 节点各司其职，通过 `MultiConnector` 组合 KV Transfer 与 KV Pool 能力，并通过 `AscendStoreConnector` 对接 Yuanrong 外部 KV 缓存池。

> 本教程默认使用 `GLM-5.1-w8a8` 权重和 A3 镜像 `quay.io/ascend/vllm-ascend:v0.18.0rc1-a3`。如果镜像中已包含 GLM-5.1 和 Yuanrong 所需依赖，可跳过对应安装步骤。
>
> **当前版本补丁要求**：在当前版本下使用 Yuanrong 多级缓存前，需要先获取单独提供的以下三个 patch 文件。建议先将 patch 上传到容器内固定目录 `/workspace/yuanrong_patches/`，然后依次执行：
>
> ```bash
> mkdir -p /workspace/yuanrong_patches
> cd /vllm-workspace/vllm
> git am /workspace/yuanrong_patches/0001-Bugfix-Fix-negative-local_cache_hit-in-P-D-disaggreg.patch
>
> cd /vllm-workspace/vllm-ascend
> git am /workspace/yuanrong_patches/0001-Implement-yuanrong-backend.patch
> git am /workspace/yuanrong_patches/0001-BugFix-0.18.0-KV-Pool-Fix-KV-Pool-not-putting-kv-cac.patch
> ```
>
> 这三个 patch 文件需要单独提供，并上传到上述目录。其中，`0001-Bugfix-Fix-negative-local_cache_hit-in-P-D-disaggreg.patch` 需要打到 `/vllm-workspace/vllm` 仓库下，用于修复 `local_cache_hit` 指标出现负值的问题；`0001-Implement-yuanrong-backend.patch` 用于补充 Yuanrong backend 支持；`0001-BugFix-0.18.0-KV-Pool-Fix-KV-Pool-not-putting-kv-cac.patch` 用于修复 vLLM v0.18.0 在 speculative decoding 场景下 KV Pool 未正确执行 KV Cache put / finalize 的问题，并规避后续 vLLM metrics 统计相关报错。若环境中已包含这些 patch 的改动，可跳过此步骤。
>
> **W8A8 说明**：A3 上 `VLLM_ASCEND_ENABLE_FUSED_MC2=1` 可以开启 Prefill/Decode 两侧的 MoE 融合算子。这一能力当前仅支持 `W8A8`，仍属于实验特性；如遇稳定性问题，可先回退为 `VLLM_ASCEND_ENABLE_FUSED_MC2=0`。

## 环境准备

### 硬件要求

- 4 台 Atlas 800I A3 或 Atlas 800T A3 服务器。
- 每台服务器提供 16 个 NPU device。
- 4 台服务器之间已完成灵衢组网，业务网卡和 NPU 通信网络互通。
- 服务器时间建议保持一致。

### 软件要求

| 组件 | 版本建议 | 说明 |
|------|----------|------|
| vLLM Ascend 镜像 | `v0.18.0rc1-a3` | A3 Ubuntu 镜像 |
| CANN | `>= 8.5.0` | 以镜像配套版本为准 |
| HDK | `>= 25.5.0` | A3 多机建议使用配套 HDK |
| Python 依赖 | `transformers==5.4.0` | GLM-5.1 需要较新版本 |

若希望在 A3 上启用 `ASCEND_ENABLE_USE_FABRIC_MEM=1`，建议满足 `HDK >= 26.0.0` 且 `CANN >= 9.0.0`。

### 网络检查

在每个节点上检查 NPU 状态：

```bash
npu-smi info
```

获取各 NPU 的 vNIC 信息：

```bash
for i in {0..15}; do hccn_tool -i $i -vnic -g; done
```

在跨节点场景下，使用对端 NPU IP 检查连通性：

```bash
for i in {0..15}; do hccn_tool -i $i -hccs_ping -g address <对端NPU_IP>; done
```

所有关键链路应处于可用状态后再启动推理服务。

### 模型权重

下载 GLM-5.1 W8A8 权重并放置到所有节点可访问的相同路径，例如：

```bash
/data/GLM-5.1-w8a8
```

模型下载地址：[魔塔社区](https://modelscope.cn/models/Eco-Tech/GLM-5.1-w8a8)。

## 使用 Docker 运行

拉取 A3 镜像：

```bash
docker pull quay.io/ascend/vllm-ascend:v0.18.0rc1-a3
```

在所有 4 个节点保存同一份 `start-docker.sh`：

```bash
#!/bin/bash
IMAGES_ID="$1"
NAME="$2"
SHM_SIZE="$3"

if [ $# -ne 3 ]; then
    echo "error: 需要传入 3 个参数，格式：$0 <镜像ID> <容器名> <共享内存大小>"
    exit 1
fi

if ! docker images --format "{{.ID}}" | grep -q "^${IMAGES_ID:0:12}$"; then
    echo "error: 镜像ID ${IMAGES_ID} 不存在"
    exit 1
fi

docker run --name "${NAME}" -it -d --net=host --shm-size="${SHM_SIZE}" \
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

创建并进入容器：

```bash
docker images | grep vllm-ascend

# Prefill 节点（节点 0、节点 1）：500 GB
bash start-docker.sh <镜像ID> glm51-a3-large-ep-p 500g
docker exec -it glm51-a3-large-ep-p bash

# Decode 节点（节点 2、节点 3）：100 GB
bash start-docker.sh <镜像ID> glm51-a3-large-ep-d 100g
docker exec -it glm51-a3-large-ep-d bash
```

进入容器后，如果镜像内 `transformers` 版本低于 GLM-5.1 要求，执行：

```bash
pip install --upgrade transformers==5.4.0
```

## 安装 Yuanrong Datasystem

在所有 4 个节点的容器内安装 Yuanrong Datasystem：

```bash
wget https://gitcode.com/openeuler/yuanrong-datasystem/releases/download/v0.7.6.rc1/openyuanrong_datasystem-0.7.6rc1-cp311-cp311-manylinux_2_35_aarch64.whl
pip install openyuanrong_datasystem-0.7.6rc1-cp311-cp311-manylinux_2_35_aarch64.whl
```

验证安装：

```bash
python -c "import yr.datasystem; print('Yuanrong Datasystem 安装成功')"
```

## 安装 etcd

Yuanrong 服务启动脚本依赖 `etcd` 和 `etcdctl`。4 机场景建议仅在节点 0 安装并启动 etcd，其他节点复用同一个 etcd。

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

## 节点规划

本文使用以下示例 IP，请按实际环境替换：

| 节点 | 角色 | 示例 IP | 本地服务端口 |
|------|------|---------|--------------|
| 节点 0 | Prefill 主节点 | `192.168.10.10` | `6600`、`6601` |
| 节点 1 | Prefill 从节点 | `192.168.10.11` | `6600`、`6601` |
| 节点 2 | Decode 主节点 | `192.168.10.20` | `6700`、`6701`、`6702`、`6703` |
| 节点 3 | Decode 从节点 | `192.168.10.21` | `6700`、`6701`、`6702`、`6703` |

### 内存规划

| 分组 | Docker `--shm-size` | Yuanrong `SHM_SIZE` | 说明 |
|------|---------------------|---------------------|------|
| Prefill 节点 | `500g` | `512000` | 节点 0、节点 1 |
| Decode 节点 | `100g` | `102400` | 节点 2、节点 3 |

`SHM_SIZE` 单位为 MB，会传给 `dscli start -w --shared_memory_size_mb`。Docker `--shm-size` 不应小于 Yuanrong Worker 的共享内存配置。

### 端口规划

| 用途 | 示例端口 | 说明 |
|------|----------|------|
| Prefill DP RPC | `10521` | Prefill 组 DP 通信端口 |
| Decode DP RPC | `10523` | Decode 组 DP 通信端口 |
| Prefill vLLM | `6600-6601` | 每个 Prefill 节点 2 个本地 DP 实例 |
| Decode vLLM | `6700-6703` | 每个 Decode 节点 4 个本地 DP 实例 |
| 代理服务 | `8000` | 对外 OpenAI API 入口 |
| etcd | `2379` | Yuanrong 元数据服务 |
| Yuanrong Worker | `18481` | 每个节点启动一个 Worker |
| Prefill KV Transfer | `36000` | Mooncake P 侧端口 |
| Decode KV Transfer | `36200` | Mooncake D 侧端口 |

> A3 16 device 场景中，Mooncake AscendDirectTransport 可能占用 `[20000, 35999]` 区间内端口。本文将 `kv_port` 设置为 `36000` 和 `36200`，避免与该保留区间冲突。

## 启动 Yuanrong 服务

所有节点都需要启动 Yuanrong Worker，并连接到节点 0 上的 etcd。

### 节点 0：Prefill 主节点

创建 `run_yr_node0.sh`：

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

etcdctl --endpoints "${ETCD_IP}:2379" endpoint health

dscli start -w \
  --worker_address ${HOST_IP}:${WORKER_PORT} \
  --etcd_address ${ETCD_IP}:${ETCD_PORT} \
  --shared_memory_size_mb ${SHM_SIZE} \
  --node_timeout_s ${NODE_TIMEOUT} \
  --node_dead_timeout_s ${NODE_DEAD_TIMEOUT} \
  --liveness_check_path ${LIVENESS_PATH}
```

### 节点 1：Prefill 从节点

创建 `run_yr_worker.sh`：

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

dscli start -w \
  --worker_address ${HOST_IP}:${WORKER_PORT} \
  --etcd_address ${ETCD_IP}:${ETCD_PORT} \
  --shared_memory_size_mb ${SHM_SIZE} \
  --node_timeout_s ${NODE_TIMEOUT} \
  --node_dead_timeout_s ${NODE_DEAD_TIMEOUT} \
  --liveness_check_path ${LIVENESS_PATH}
```

### 节点 2-3：Decode 节点

分别创建 `run_yr_worker.sh`。`HOST_IP` 修改为当前 Decode 节点 IP，`ETCD_IP` 固定为节点 0 IP。

```bash
#!/bin/bash

export HOST_IP="<当前Decode节点IP>"
export ETCD_IP="192.168.10.10"
export WORKER_PORT=18481
export ETCD_PORT=2379
export SHM_SIZE=102400
export NODE_TIMEOUT=30
export NODE_DEAD_TIMEOUT=60
export LIVENESS_PATH=/workspace/liveness

dscli start -w \
  --worker_address ${HOST_IP}:${WORKER_PORT} \
  --etcd_address ${ETCD_IP}:${ETCD_PORT} \
  --shared_memory_size_mb ${SHM_SIZE} \
  --node_timeout_s ${NODE_TIMEOUT} \
  --node_dead_timeout_s ${NODE_DEAD_TIMEOUT} \
  --liveness_check_path ${LIVENESS_PATH}
```

## 准备部署脚本

在所有节点进入容器后创建工作目录，并复制外部 DP 启动脚本和代理脚本：

```bash
mkdir -p /workspace/glm51-a3-large-ep
cd /workspace/glm51-a3-large-ep

cp /vllm-workspace/vllm-ascend/examples/external_online_dp/launch_online_dp.py .
cp /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py .
```

`launch_online_dp.py` 会在当前目录查找 `./run_dp_template.sh`，因此后续命令都应在 `/workspace/glm51-a3-large-ep` 下执行。

## Prefill 节点配置

节点 0 和节点 1 都创建 `run_dp_template.sh`。两个节点的脚本内容一致，只需修改 `nic_name`、`local_ip` 和 `MODEL_PATH`。

```bash
#!/bin/bash

nic_name="enp67s0f0np0"
local_ip="192.168.10.10"
MODEL_PATH="/data/GLM-5.1-w8a8"

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256
export ASCEND_AGGREGATE_ENABLE=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export ASCEND_BUFFER_POOL=4:8
export PYTHONHASHSEED=0
export VLLM_ENGINE_READY_TIMEOUT_S=1800

export DS_WORKER_ADDR="${local_ip}:18481"
export DS_H2D_MEMCPY_POLICY="direct"
export DS_D2H_MEMCPY_POLICY="direct"
unset GOOGLE_LOGTOSTDERR GOOGLE_ALSOLOGTOSTDERR

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
                        "dp_size": 4,
                        "tp_size": 8
                    },
                    "decode": {
                        "dp_size": 8,
                        "tp_size": 4
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
    --served-model-name glm-5.1 \
    --max-model-len 202752 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --additional-config '{"enable_npugraph_ex": true, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    2>&1 | tee "./glm51_prefill_dp${4}.log"
```

为两个 Prefill 节点分别创建 `server.sh`。

节点 0：

```bash
#!/bin/bash
cd /workspace/glm51-a3-large-ep

python launch_online_dp.py \
    --dp-size 4 \
    --tp-size 8 \
    --dp-size-local 2 \
    --dp-rank-start 0 \
    --dp-address 192.168.10.10 \
    --dp-rpc-port 10521 \
    --vllm-start-port 6600
```

节点 1：

```bash
#!/bin/bash
cd /workspace/glm51-a3-large-ep

python launch_online_dp.py \
    --dp-size 4 \
    --tp-size 8 \
    --dp-size-local 2 \
    --dp-rank-start 2 \
    --dp-address 192.168.10.10 \
    --dp-rpc-port 10521 \
    --vllm-start-port 6600
```

## Decode 节点配置

节点 2 和节点 3 都创建 `run_dp_template.sh`。两个节点的脚本内容一致，只需修改 `nic_name`、`local_ip` 和 `MODEL_PATH`。

```bash
#!/bin/bash

nic_name="enp67s0f0np0"
local_ip="192.168.10.20"
MODEL_PATH="/data/GLM-5.1-w8a8"

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export ASCEND_BUFFER_POOL=4:8
export PYTHONHASHSEED=0
export VLLM_ENGINE_READY_TIMEOUT_S=1800

export DS_WORKER_ADDR="${local_ip}:18481"
export DS_H2D_MEMCPY_POLICY="direct"
export DS_D2H_MEMCPY_POLICY="direct"
unset GOOGLE_LOGTOSTDERR GOOGLE_ALSOLOGTOSTDERR

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
                        "dp_size": 4,
                        "tp_size": 8
                    },
                    "decode": {
                        "dp_size": 8,
                        "tp_size": 4
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
    --served-model-name glm-5.1 \
    --max-model-len 202752 \
    --max-num-batched-tokens 32 \
    --trust-remote-code \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.92 \
    --async-scheduling \
    --quantization ascend \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 12, 16, 20, 24, 28, 32]}' \
    --additional-config '{"enable_npugraph_ex": true, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    2>&1 | tee "./glm51_decode_dp${4}.log"
```

为两个 Decode 节点分别创建 `server.sh`。

节点 2：

```bash
#!/bin/bash
cd /workspace/glm51-a3-large-ep

python launch_online_dp.py \
    --dp-size 8 \
    --tp-size 4 \
    --dp-size-local 4 \
    --dp-rank-start 0 \
    --dp-address 192.168.10.20 \
    --dp-rpc-port 10523 \
    --vllm-start-port 6700
```

节点 3：

```bash
#!/bin/bash
cd /workspace/glm51-a3-large-ep

python launch_online_dp.py \
    --dp-size 8 \
    --tp-size 4 \
    --dp-size-local 4 \
    --dp-rank-start 4 \
    --dp-address 192.168.10.20 \
    --dp-rpc-port 10523 \
    --vllm-start-port 6700
```

## 配置代理

在节点 0 创建 `proxy.sh`：

```bash
#!/bin/bash
cd /workspace/glm51-a3-large-ep

unset http_proxy
unset https_proxy

python load_balance_proxy_server_example.py \
    --host 0.0.0.0 \
    --port 8000 \
    --prefiller-hosts \
        192.168.10.10 \
        192.168.10.10 \
        192.168.10.11 \
        192.168.10.11 \
    --prefiller-ports \
        6600 \
        6601 \
        6600 \
        6601 \
    --decoder-hosts \
        192.168.10.20 \
        192.168.10.20 \
        192.168.10.20 \
        192.168.10.20 \
        192.168.10.21 \
        192.168.10.21 \
        192.168.10.21 \
        192.168.10.21 \
    --decoder-ports \
        6700 \
        6701 \
        6702 \
        6703 \
        6700 \
        6701 \
        6702 \
        6703
```

## 启动顺序

建议按以下顺序启动：

```bash
# 1. 节点 0：启动 etcd 和 Yuanrong Worker
bash run_yr_node0.sh

# 2. 节点 1：启动 Prefill Yuanrong Worker，SHM_SIZE=512000
bash run_yr_worker.sh

# 3. 节点 2、节点 3：启动 Decode Yuanrong Worker，SHM_SIZE=102400
bash run_yr_worker.sh

# 4. 节点 0、节点 1：启动 Prefill 实例
bash server.sh

# 5. 节点 2、节点 3：启动 Decode 实例
bash server.sh

# 6. 等所有后端日志出现 Application startup complete 后，在节点 0 启动代理
bash proxy.sh > proxy.log 2>&1 &
```

检查日志：

```bash
tail -f glm51_prefill_dp0.log
tail -f glm51_decode_dp0.log
tail -f proxy.log
```

后端日志中出现 `Application startup complete` 表示对应实例启动完成。

## 功能验证

### 健康检查

```bash
curl http://192.168.10.10:8000/healthcheck
```

### 推理请求

```bash
curl -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -X POST \
    -d '{
        "model": "glm-5.1",
        "messages": [{
            "role": "user",
            "content": "请介绍一下 GLM-5.1 在软件工程场景中的典型用法。"
        }],
        "stream": false,
        "temperature": 0,
        "max_tokens": 256
    }' http://192.168.10.10:8000/v1/chat/completions
```

### 查看 metrics

代理不会聚合各后端的 metrics，建议直接查看具体后端实例：

```bash
curl http://192.168.10.20:6700/metrics | head
```

### 缓存命中率监控

推荐直接查看后端实例日志，而不是代理日志。本文示例日志文件名如下：

- Prefill 节点：`glm51_prefill_dp0.log`、`glm51_prefill_dp1.log`、`glm51_prefill_dp2.log`、`glm51_prefill_dp3.log`
- Decode 节点：`glm51_decode_dp0.log` 至 `glm51_decode_dp7.log`

```bash
tail -f glm51_decode_dp0.log
tail -f glm51_decode_dp0.log | grep -E "Prefix cache hit rate|External prefix cache hit rate|num_computed_tokens"
```

也可以直接查看后端 metrics：

```bash
curl http://192.168.10.20:6700/metrics | grep external_prefix_cache
```

如果当前环境包含 `vllm-ascend` 仓库源码，也可使用仓库脚本持续观察：

```bash
bash tools/watch_cache_hit_rate.sh -u http://192.168.10.20:6700/metrics -i 10
```

## 关键参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--data-parallel-size` | P=`4`，D=`8` | P/D 两组分别独立配置全局 DP |
| `--tensor-parallel-size` | P=`8`，D=`4` | 每个 DP 副本占用的 NPU 数 |
| `--enable-expert-parallel` | 开启 | GLM-5.1 MoE 专家并行 |
| `--max-model-len` | `202752` | 参考 GLM-5.1 A3 长上下文配置 |
| `--max-num-batched-tokens` | P=`4096`，D=`32` | P 侧偏向预填充吞吐，D 侧偏向 decode 延迟 |
| `--async-scheduling` | D 侧开启 | Decode 侧开启异步调度 |
| `--compilation-config` | D 侧 `FULL_DECODE_ONLY` | Decode 侧图模式配置 |
| `kv_connector` | `MultiConnector` | 组合 KV Transfer 与 Yuanrong KV Pool |
| `kv_port` | P=`36000`，D=`36200` | 避开 A3 16 device 保留端口区间 |
| `backend` | `yuanrong` | `AscendStoreConnector` 使用 Yuanrong 后端 |
| `lookup_rpc_port` | `$4` | 使用 DP rank 区分同机不同实例 |
| `engine_id` | `$4` | 建议与 DP rank 保持一致，便于排障 |
| `PYTHONHASHSEED` | `0` | 所有节点保持一致 |
| `DS_WORKER_ADDR` | `${local_ip}:18481` | 当前节点 Yuanrong Worker 地址 |
| Docker `--shm-size` | P=`500g`，D=`100g` | 容器共享内存大小 |
| Yuanrong `SHM_SIZE` | P=`512000`，D=`102400` | `--shared_memory_size_mb`，单位 MB |
| `ASCEND_ENABLE_USE_FABRIC_MEM` | 可选 `1` | A3 统一地址直传，需满足版本要求 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `1` | W8A8 可开启融合 MC2；如遇稳定性问题可回退为 `0` |

## MultiConnector 配置要点

本文使用 `MultiConnector` 组合两类能力：

- `MooncakeConnectorV1`：负责 P/D 节点间的 KV Transfer。
- `AscendStoreConnector`：负责接入 Yuanrong 外部 KV Pool。

配置时需要注意：

- 顶层 `kv_role` 与子连接器 `kv_role` 必须一致，P 侧为 `kv_producer`，D 侧为 `kv_consumer`。
- P/D 两侧的 `prefill.dp_size/tp_size` 和 `decode.dp_size/tp_size` 必须与实际启动参数一致。
- 同一台机器上不同 DP 副本的 `lookup_rpc_port` 必须唯一。本文直接使用 `$4`，即当前 DP rank。
- 所有节点必须设置相同的 `PYTHONHASHSEED=0`，确保 KV Cache 键计算一致。
- A3 场景不要继续沿用 A2 文档中的 `30000/30100` 端口组合，建议使用本文示例中的 `36000/36200`。

## 故障排查

1. **`zmq.error.ZMQError: Address already in use`**

   检查 `kv_port` 是否落在 `[20000, 35999]`。A3 16 device 场景建议使用本文示例的 `36000` 和 `36200`。

2. **DP 组无法建立连接**

   检查 `--data-parallel-address` 是否填写对应组的主节点业务 IP。Prefill 组应使用节点 0 IP，Decode 组应使用节点 2 IP。

3. **HCCL 或 Gloo 通信失败**

   检查 `nic_name`、`local_ip`、`HCCL_SOCKET_IFNAME`、`GLOO_SOCKET_IFNAME` 是否与实际网卡一致，并确认防火墙放行 DP RPC 和后端服务端口。

4. **启用 `VLLM_ASCEND_ENABLE_FUSED_MC2=1` 后启动失败**

   该融合路径面向 W8A8。可先回退：

   ```bash
   export VLLM_ASCEND_ENABLE_FUSED_MC2=0
   ```

   如果回退后恢复正常，再继续排查融合算子相关问题。

5. **代理健康检查失败**

   先确认所有后端实例已启动，再检查 `proxy.sh` 中的 host 和 port 数量是否与实际实例一致。本文应配置 4 个 Prefill 实例和 8 个 Decode 实例。

6. **推理请求返回模型不存在**

   请求中的 `model` 字段需要与启动参数 `--served-model-name glm-5.1` 保持一致。

7. **Yuanrong Worker 注册失败**

   检查当前节点 Worker 是否监听，以及 `DS_WORKER_ADDR` 是否与 `dscli start -w` 的 `--worker_address` 一致：

   ```bash
   netstat -tlnp | grep 18481
   echo $DS_WORKER_ADDR
   ```

8. **etcd 连接失败**

   在节点 0 检查 etcd 健康状态，并确认其他节点可以访问节点 0 的 `2379` 端口：

   ```bash
   etcdctl --endpoints "192.168.10.10:2379" endpoint health
   ```

## 参考

- [vLLM Ascend 安装文档](../../installation.md#set-up-using-docker)
- [GLM-5.1 昇腾支持汇总文档](glm-5.1-ascend-support.md)
- [Disaggregated Prefill 开发指南](../../developer_guide/feature_guide/disaggregated_prefill.md)
