# 基于 Yuanrong 的 GLM-5 W8A8 8机 A2 大EP PD分离部署

## 概述

本指南提供在 8 台 Atlas 800I A2 服务器上部署 GLM-5 W8A8 模型，使用 PD 分离架构（1P1D）并叠加 Yuanrong Datasystem 作为 KV Pool 后端的详细步骤。

PD 分离架构下，Prefill 节点与 Decode 节点各司其职，通过 MultiConnector 组合 KV Transfer 与 KV Pool 能力，同时通过 AscendStoreConnector（Yuanrong 后端）实现外部 KV 缓存池，支持前缀缓存复用，降低重复前缀场景下的首 token 时延。

> **前置条件**：本教程使用的 Docker 镜像版本为 `vllm-ascend:0.18.0rc1`。如本地尚未下载，请先参考下文"使用 Docker 运行"中的命令拉取对应镜像。
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

- **8 × Atlas 800I A2 服务器**，每台配备 8 张 NPU 卡（每张 64G 显存）
- 已配置 RoCE 网络以获得最佳性能

### 软件要求

采用模型配套的 Docker 镜像，软件版本与 Docker 镜像内置版本保持一致，确保 HDK、固件等软件在配套范围内。
此外：CANN 版本要求至少高于 8.5.0，HDK 版本要求至少高于 25.2.3（启用 RH2D 通过 RoCE 传输时，HDK 版本需要 25.5.0 以上）。

### 环境信息

| 组件 | 版本 | 备注 |
|------|------|------|
| 服务器硬件 | Atlas 800I A2 × 8 | 4机P + 4机D |
| vLLM-Ascend | vllm-ascend:0.18.0rc1 | |
| CANN | ≥ 8.5.0 | |
| HDK | ≥ 25.2.3（启用 RH2D 需 ≥ 25.5.0） | RH2D 通过 RoCE 需要 HDK ≥ 25.5.0 |
| GLM-5 权重 | W8A8 量化 | [ModelScope](https://modelscope.cn/models/umiiiiii/GLM-W8A8/files) |

### 模型权重

下载 GLM-5 W8A8 模型权重并放置到指定目录，如 `/home/models/GLM-W8A8/`。

> **模型下载地址**：[魔搭社区](https://modelscope.cn/models/umiiiiii/GLM-W8A8/files)

## 使用 Docker 运行

本教程使用的 Docker 镜像版本为 `vllm-ascend:0.18.0rc1`。如本地尚未下载，可执行以下命令：

```bash
docker pull quay.io/ascend/vllm-ascend:0.18.0rc1
```

如果下载较慢，可将 `quay.io` 替换为 `m.daocloud.io/quay.io` 或 `quay.nju.edu.cn` 以加速拉取。更多镜像说明可参考[安装文档](../../installation.md#set-up-using-docker)。

### 创建容器

在所有 8 个节点上分别保存同一份 `start-docker.sh`：

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

在每个节点分别创建容器：

```bash
# 在每个节点执行（替换为实际镜像ID）
bash start-docker.sh <镜像ID> glm5-pd-yuanrong
```

进入容器：

```bash
docker exec -it glm5-pd-yuanrong bash
```

### 升级 transformers 版本

GLM-5 模型要求较高版本的 transformers，进入容器后需先升级：

```bash
pip install transformers==5.2.0 --no-deps --force-reinstall
pip install huggingface_hub==1.5.0 --no-deps --force-reinstall
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

后续 Yuanrong 服务启动脚本依赖 `etcd` 和 `etcdctl`。至少在 P 主节点安装。

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

8 机场景需要在所有 8 个节点都启动 Datasystem Worker，并连接同一个 etcd。

### 节点 0（P 主节点）

创建启动脚本 `run_yr_node0.sh`（启动 etcd 和 Worker）：

```bash
#!/bin/bash

export HOST_IP="<P主节点IP>"
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

etcdctl --endpoints "${ETCD_IP}:2379" put key "value"
etcdctl --endpoints "${ETCD_IP}:2379" get key

dscli start -w \
  --worker_address ${HOST_IP}:${WORKER_PORT} \
  --etcd_address ${ETCD_IP}:${ETCD_PORT} \
  --shared_memory_size_mb ${SHM_SIZE} \
  --node_timeout_s ${NODE_TIMEOUT} \
  --node_dead_timeout_s ${NODE_DEAD_TIMEOUT} \
  --liveness_check_path ${LIVENESS_PATH}

echo "节点 0 Yuanrong 服务启动完成"
echo "etcd 日志: /tmp/etcd.log"
```

### 其他节点（1-7）

每个节点创建启动脚本 `run_yr_worker.sh`（只启动 Worker，连接节点 0 的 etcd）：

```bash
#!/bin/bash

export HOST_IP="<当前节点IP>"
export ETCD_IP="<P主节点IP>"
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

echo "当前节点 Yuanrong 服务启动完成"
```

运行顺序：

```bash
# 节点 0 先启动
bash run_yr_node0.sh

# 其他节点后启动
bash run_yr_worker.sh
```

### etcd 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| name | etcd-single | etcd 节点名称，集群中必须唯一 |
| data-dir | /tmp/etcd-data | 数据存储目录，用于持久化保存 etcd 数据 |
| listen-client-urls | http://0.0.0.0:2379 | 监听客户端请求的 URL 地址 |
| advertise-client-urls | http://${ETCD_IP}:2379 | 对外广播的客户端 URL |
| listen-peer-urls | http://0.0.0.0:2380 | 监听集群节点间通信的 URL 地址 |
| initial-advertise-peer-urls | http://${ETCD_IP}:2380 | 对外广播的集群通信 URL |
| initial-cluster | etcd-single=http://${ETCD_IP}:2380 | 初始集群配置 |

> **参考文档**：[etcd 官方文档](https://etcd.io/docs/)
>
> **生产环境建议**：上述示例为单实例部署，适用于测试和开发环境。对于可靠性要求较高的生产环境，建议部署 etcd 集群（通常 3 或 5 个节点）。

### Datasystem Worker 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| worker_address | ${HOST_IP}:${WORKER_PORT} | Worker 服务地址和端口 |
| etcd_address | ${ETCD_IP}:${ETCD_PORT} | etcd 服务发现地址 |
| shared_memory_size_mb | 512000 | 共享内存大小（500 GB） |
| node_timeout_s | 30 | 节点超时时间（秒） |
| node_dead_timeout_s | 60 | 节点死亡超时时间（秒） |
| liveness_check_path | /workspace/liveness | 存活检查路径 |

> **参考文档**：[Yuanrong Datasystem 文档](https://atomgit.com/openeuler/yuanrong-datasystem)

停止 Worker：

```bash
dscli stop --worker_address ${HOST_IP}:${WORKER_PORT}
```

## PD分离部署（8机、1P1D + Yuanrong）

### 并行策略

- P节点：DP4，TP8（4机，每机 1 个数据并行副本）
- D节点：DP8，TP4（4机，每机 2 个数据并行副本）

### 节点分配

| 节点 | 角色 | IP（示例） | 需要文件 |
|------|------|------------|----------|
| 节点 0 | P 主节点 | 71.10.29.138 | launch_online_dp.py、run_dp_template.sh、server.sh、proxy.sh、load_balance_proxy_server_example.py |
| 节点 1 | P 从节点 | 71.10.29.141 | launch_online_dp.py、run_dp_template.sh、server.sh |
| 节点 2 | P 从节点 | 71.10.29.125 | launch_online_dp.py、run_dp_template.sh、server.sh |
| 节点 3 | P 从节点 | 71.10.29.128 | launch_online_dp.py、run_dp_template.sh、server.sh |
| 节点 4 | D 主节点 | 71.10.29.124 | launch_online_dp.py、run_dp_template.sh、server.sh |
| 节点 5 | D 从节点 | 71.10.29.123 | launch_online_dp.py、run_dp_template.sh、server.sh |
| 节点 6 | D 从节点 | 71.10.29.139 | launch_online_dp.py、run_dp_template.sh、server.sh |
| 节点 7 | D 从节点 | 71.10.29.142 | launch_online_dp.py、run_dp_template.sh、server.sh |

**脚本说明**：
- [`launch_online_dp.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py)：每个节点都要有，无需修改
- [`run_dp_template.sh`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/run_dp_template.sh)：每个节点根据实际情况修改
- [`dp_load_balance_proxy_server.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/dp_load_balance_proxy_server.py)：仅 P 主节点需要

详细说明见：[external_online_dp README](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/README.md)

### P节点

`run_dp_template.sh` 模板，请按实际情况修改 `nic_name`、`local_ip`、权重路径、`rot_path`：

```bash
#!/bin/bash

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

# Yuanrong Datasystem
export DS_WORKER_ADDR="${local_ip}:18481"
export DS_H2D_MEMCPY_POLICY="direct"
export DS_D2H_MEMCPY_POLICY="direct"
unset GOOGLE_LOGTOSTDERR GOOGLE_ALSOLOGTOSTDERR

# vLLM
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export PYTHONHASHSEED=0

vllm serve /home/models/GLM-W8A8 \
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
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "engine_id": "0",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnectorV1",
                    "kv_role": "kv_producer",
                    "kv_port": "30000",
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
                        "lookup_rpc_port": "0",
                        "backend": "yuanrong"
                    }
                }
            ]
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
    2>&1 | tee glm.log
```

**server.sh**：P节点 DP4、TP8

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

**proxy.sh**：只存在于 P 主节点，在 P/D 节点服务启动成功后执行 `bash proxy.sh > proxy.log &`，根据实际情况修改组网 IP。

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

### D节点

`run_dp_template.sh` 模板，请按实际情况修改 `nic_name`、`local_ip`、权重路径、`rot_path`：

```bash
#!/bin/bash

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

# Yuanrong Datasystem
export DS_WORKER_ADDR="${local_ip}:18481"
export DS_H2D_MEMCPY_POLICY="direct"
export DS_D2H_MEMCPY_POLICY="direct"
unset GOOGLE_LOGTOSTDERR GOOGLE_ALSOLOGTOSTDERR

# vLLM
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export PYTHONHASHSEED=0

vllm serve /home/models/GLM-W8A8 \
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
    "{
        \"kv_connector\": \"MultiConnector\",
        \"kv_role\": \"kv_consumer\",
        \"kv_connector_extra_config\": {
            \"connectors\": [
                {
                    \"kv_connector\": \"MooncakeConnectorV1\",
                    \"kv_role\": \"kv_consumer\",
                    \"kv_port\": \"30100\",
                    \"kv_connector_module_path\": \"vllm_ascend.distributed.mooncake_connector\",
                    \"kv_connector_extra_config\": {
                        \"use_ascend_direct\": true,
                        \"prefill\": {
                            \"dp_size\": 4,
                            \"tp_size\": 8
                        },
                        \"decode\": {
                            \"dp_size\": 8,
                            \"tp_size\": 4
                        }
                    }
                },
                {
                    \"kv_connector\": \"AscendStoreConnector\",
                    \"kv_role\": \"kv_consumer\",
                    \"kv_connector_extra_config\": {
                        \"lookup_rpc_port\": \"$4\",
                        \"backend\": \"yuanrong\"
                    }
                }
            ]
        }
    }" \
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
    2>&1 | tee glm.log
```

**server.sh**：D节点 DP8、TP4

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

### 配置参数说明

#### P节点参数

| 参数 | 值 | 说明 |
|------|-----|------|
| tensor-parallel-size | 8 | 每节点使用 8 张 NPU 卡 |
| data-parallel-size | 4 | 数据并行大小（4 个 P 节点） |
| max-model-len | 133120 | 最大上下文长度 |
| max-num-batched-tokens | 4096 | 最大批处理 token 数 |
| max-num-seqs | 32 | 最大并发序列数 |
| gpu-memory-utilization | 0.95 | GPU 显存利用率 |
| quantization | ascend | 使用 Ascend 量化 |
| enable-expert-parallel | (标志) | 启用 MoE 专家并行 |
| enable-chunked-prefill | (标志) | 启用分块预填充 |
| async-scheduling | (标志) | 启用异步调度 |
| kv_connector | MultiConnector | 使用多连接器组合 |
| kv_role (P节点) | kv_producer | Prefill 节点作为 KV 生产者 |
| MooncakeConnectorV1 kv_port | 30000 | Mooncake 传输端口 |
| AscendStoreConnector backend | yuanrong | 使用 Yuanrong 后端 |
| AscendStoreConnector lookup_rpc_port (P节点) | 0 | P节点每机仅1个数据并行副本，固定为0 |
| AscendStoreConnector lookup_rpc_port (D节点) | $4 (dp_rank) | D节点每机2个数据并行副本，使用dp_rank自动区分 |

#### D节点参数

| 参数 | 值 | 说明 |
|------|-----|------|
| tensor-parallel-size | 4 | 每个数据并行副本使用 4 张 NPU 卡 |
| data-parallel-size | 8 | 数据并行大小（8 个数据并行副本） |
| max-model-len | 133120 | 最大上下文长度 |
| max-num-batched-tokens | 32 | 最大批处理 token 数 |
| max-num-seqs | 32 | 最大并发序列数 |
| gpu-memory-utilization | 0.95 | GPU 显存利用率 |
| kv_connector | MultiConnector | 使用多连接器组合 |
| kv_role (D节点) | kv_consumer | Decode 节点作为 KV 消费者 |
| MooncakeConnectorV1 kv_port | 30100 | Mooncake 传输端口 |
| AscendStoreConnector backend | yuanrong | 使用 Yuanrong 后端 |
| cudagraph_mode | FULL_DECODE_ONLY | 仅 Decode 阶段使用 CUDA Graph |

### 环境变量说明

| 环境变量 | 值 | 说明 |
|----------|-----|------|
| `HCCL_OP_EXPANSION_MODE` | AIV | HCCL 算子扩展模式（AI Vector 优化） |
| `OMP_PROC_BIND` | false | OpenMP 线程绑定配置 |
| `OMP_NUM_THREADS` | 10 | OpenMP 线程数 |
| `HCCL_BUFFSIZE` | 200 | HCCL 缓冲区大小 |
| `PYTORCH_NPU_ALLOC_CONF` | expandable_segments:True | NPU 显存分配策略（减少碎片） |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | 1 | 启用平衡调度 |
| `VLLM_USE_V1` | 1 | 启用 vLLM v1 架构 |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 1800 | 引擎就绪超时时间（秒） |
| `PYTHONHASHSEED` | 0 | Python 哈希种子，确保 KV Cache 键一致性 |
| `DS_WORKER_ADDR` | ${local_ip}:18481 | Yuanrong Worker 地址，必须与当前节点 dscli 启动参数一致 |
| `DS_H2D_MEMCPY_POLICY` | direct | Host-to-Device 内存拷贝策略 |
| `DS_D2H_MEMCPY_POLICY` | direct | Device-to-Host 内存拷贝策略 |
| `VLLM_ASCEND_ENABLE_MLAPO` | 1 | 启用 MLAPO 算子 |
| `VLLM_ASCEND_ENABLE_NZ` | 1 | 启用 NZ 格式 |
| `TASK_QUEUE_ENABLE` | 1 | 启用任务队列（流水优化） |
| `CPU_AFFINITY_CONF` | 1 | 启用 CPU 亲和性配置 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 1 | 启用 FLASHCOMM1 算子 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | 1 | 启用融合 MC2 |
| `ASCEND_AGGREGATE_ENABLE` | 1 | 启用聚合 |
| `ACL_OP_INIT_MODE` | 1 | ACL 算子初始化模式 |

### MultiConnector 配置结构说明

MultiConnector 的 `kv-transfer-config` JSON 结构如下：

```
{
    "kv_connector": "MultiConnector",           // 顶层使用 MultiConnector
    "kv_role": "kv_producer" | "kv_consumer",  // 顶层角色
    "engine_id": "可选，用于区分不同引擎实例",
    "kv_connector_extra_config": {
        "connectors": [                          // connectors 数组包含子连接器
            {
                "kv_connector": "MooncakeConnectorV1",   // KV Transfer（跨节点传输）
                "kv_role": "与顶层一致",
                "kv_port": "端口号",
                "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": { "dp_size": N, "tp_size": M },
                    "decode": { "dp_size": N, "tp_size": M }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",  // KV Pool（外部缓存池）
                "kv_role": "与顶层一致",
                "kv_connector_extra_config": {
                    "backend": "yuanrong",
                    "lookup_rpc_port": "端口号（同一机器上不同数据并行副本需唯一）"
                }
            }
        ]
    }
}
```

**关键注意事项**：

1. **`kv_role` 一致性**：顶层和子连接器的 `kv_role` 应保持一致（P 节点为 `kv_producer`，D 节点为 `kv_consumer`）
2. **`kv_port` 区分**：MooncakeConnectorV1 的 `kv_port` 在 Prefill 和 Decode 节点应不同（如 `30000` vs `30100`）
3. **`lookup_rpc_port` 唯一性**：AscendStoreConnector 的 `lookup_rpc_port` 在同一机器上的不同数据并行副本必须唯一
4. **`PYTHONHASHSEED`**：所有节点必须设置相同的 `PYTHONHASHSEED=0` 以保证 KV Cache 键计算一致

## 叠加特性优化

| 优化特性 | 使能方法 |
|----------|----------|
| W8A8模型量化 | [ModelScope权重](https://modelscope.cn/models/umiiiiii/GLM-W8A8/files) |
| FLASHCOMM1算子接入 | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` |
| 异步调度 | `--async-scheduling` |
| MLAPO算子接入 | `export VLLM_ASCEND_ENABLE_MLAPO=1` |
| mul_add融合算子使能 | `--additional-config` 中加 `"fuse_muls_add": true` |
| PD分离 + Yuanrong KV Pool | MultiConnector 组合 MooncakeConnectorV1 + AscendStoreConnector(yuanrong) |
| 共享专家多流 | `--additional-config` 中加 `"recompute_scheduler_enable": true` + `"multistream_overlap_shared_expert": true` |
| MTP接受率提升 | `--additional-config` 中加 `"rot_path": "xxx/rot.safetensors"` + `--speculative-config` |
| MTP-DP入图 | `"fuse_qknorm_rope": false` |
| 流水优化 | `"fuse_muls_add": true` + `export TASK_QUEUE_ENABLE=1` |
| 通信算法AIV | `"enable_npugraph_ex": true` + `export HCCL_OP_EXPANSION_MODE="AIV"` |
| FULL_DECODE_ONLY（仅D节点） | `--compilation-config` 中设置 |

## 功能验证

服务启动后，验证部署是否成功。

### 测试推理

```bash
curl -H "Accept: application/json" \
    -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "glm5",
        "messages": [{
            "role": "user",
            "content": "你好，请介绍一下人工智能的未来发展趋势。"
        }],
        "stream": false,
        "ignore_eos": false,
        "temperature": 0,
        "max_tokens": 200
    }' http://localhost:8000/v1/chat/completions
```

## 缓存命中率监控

### 查看 vLLM 日志

```bash
LOG_FILE=glm.log

# 查看最新日志
tail -f $LOG_FILE

# 实时监控命中率相关日志
tail -f $LOG_FILE | grep -E "Prefix cache hit rate|External prefix cache hit rate|num_computed_tokens"
```

### 使用脚本持续监控命中率

如果当前环境包含 `vllm-ascend` 仓库源码，可以使用仓库自带脚本持续观测命中率：

```bash
bash tools/watch_cache_hit_rate.sh -u http://localhost:8000/metrics -i 10

# 如需将结果同时保存到文件
bash tools/watch_cache_hit_rate.sh \
  -u http://localhost:8000/metrics \
  -i 10 \
  -o cache_hit_rate.log
```

常用字段：

- `local_win`：vLLM 本地 Prefix Cache 的窗口命中率
- `local_total`：vLLM 本地 Prefix Cache 的累计命中率
- `ext_win`：Yuanrong 外部 KV Cache 的窗口命中率
- `ext_total`：Yuanrong 外部 KV Cache 的累计命中率
- `eff_total`：综合本地和外部缓存后的端到端有效命中率

### 缓存命中率指标说明

| 指标 | 说明 | 统计方式 |
|------|------|----------|
| Prefix cache hit rate | **HBM（本地显存）**命中率 | **滑动窗口**：最近 1000 个请求 |
| External prefix cache hit rate | **Yuanrong（外部 KV Cache）**命中率 | **累计统计**：从服务启动到当前时刻 |
| TTFT (Time to First Token) | 首个 token 延迟 | 单次请求指标，命中率高时 TTFT 显著降低 |

**查看 Yuanrong 外部缓存命中率**：

```bash
curl http://localhost:8000/metrics | grep external_prefix_cache
```

**查看单次请求是否命中**：

```bash
grep "num_computed_tokens" $LOG_FILE
```

如果 `num_computed_tokens > 0`，表示该请求命中了缓存。

## 故障排除

### 常见问题

1. **etcd 连接失败**

   确保 etcd 正在运行且可访问：
   ```bash
   etcdctl --endpoints "${ETCD_IP}:2379" endpoint health
   ```

2. **Worker 注册失败**

   检查 Worker 地址是否正确配置：
   ```bash
   netstat -tlnp | grep 18481
   ```

3. **KV Cache 未找到**

   验证 `PYTHONHASHSEED` 设置一致：
   ```bash
   echo $PYTHONHASHSEED
   ```
   所有节点必须设置为 `0`。

4. **yr.datasystem 导入错误**

   确保 `openyuanrong-datasystem` 已安装：
   ```bash
   pip install openyuanrong-datasystem
   python -c "from yr.datasystem.hetero_client import HeteroClient; print('OK')"
   ```

5. **64 KB 页大小机器无法直接使用默认 Yuanrong 安装包**

   检查页大小：
   ```bash
   getconf PAGE_SIZE
   ```
   如果输出为 `65536`，请使用针对 64 KB 页大小单独编译的 Yuanrong 安装包。

6. **引擎启动超时**

   增加 `VLLM_ENGINE_READY_TIMEOUT_S` 的值：
   ```bash
   export VLLM_ENGINE_READY_TIMEOUT_S=3600
   ```

7. **节点时间不一致**

   8 机场景下，建议所有节点的系统时间保持一致，否则可能影响日志对齐、问题定位以及部分依赖时间戳的排障判断。
   ```bash
   date
   timedatectl
   ```

8. **transformers 版本过低**

   升级 transformer 版本：
   ```bash
   pip install transformers==5.2.0 --no-deps --force-reinstall
   pip install huggingface_hub==1.5.0 --no-deps --force-reinstall
   ```

9. **加了 rot.safetensors，报错 `KeyError: 'rot'`**

   rot 的权重要放在其他目录，不能和 W8A8 的放一起。

10. **部署上下文 200K，报超出模型最大支持长度**

    将 `--max-model-len` 参数调低，官方宣称最大支持 200K 上下文，实测最大只能到 **198K**。

11. **PP并行策略报错**

    模型不支持 PP 并行策略，改用 **DP+TP** 并行策略。

## 参考资料

- [Yuanrong Datasystem 文档](https://atomgit.com/openeuler/yuanrong-datasystem)
- [etcd 文档](https://etcd.io/docs/)
- [vLLM Ascend 文档](https://docs.vllm.ai/projects/ascend/)
- [GLM-5 W8A8 A2 部署调优实践](GLM5.md)
- [基于 Yuanrong 的 GLM-5 W4A8 单实例部署](pd_colocated_yuanrong_glm5_cn.md)
- [KV Pool 使用指南](../../user_guide/feature_guide/kv_pool.md)
