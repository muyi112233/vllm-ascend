# 基于 Mooncake 的 GLM-5 W4A8 单实例部署

## 概述

本指南提供在 Atlas 800I A2 服务器（双机）或 Atlas 800I A3 / Atlas 800T A3 服务器（单机）上部署 GLM-5 W4A8 模型，并使用
Mooncake 作为 KV Pool 后端的详细步骤。

GLM-5 是采用混合专家架构的高效推理模型，专为复杂系统工程和长时序智能体任务设计。使用 Mooncake 作为
KV Pool 后端，可以实现外部 KV Cache 存储与请求间复用，从而降低重复前缀场景下的首 token 时延。

本教程沿用 `AscendStoreConnector` 接入 KV Pool，并显式指定 `backend=mooncake`。与
`pd_colocated_yuanrong_glm5_cn.md` 相比，本教程不再依赖 Yuanrong Datasystem、`etcd` 和 `dscli`，
而是改为启动 `mooncake_master` 并通过 `mooncake.json` 完成后端配置。

> **前置条件**：本教程使用的 Docker 镜像版本为 `vllm-ascend:0.18.0rc1`。如本地尚未下载，请先参考下文“使用 Docker 运行”中的命令拉取对应镜像。
>
> **当前版本补丁要求**：在当前版本下，如果使用 KV Pool，建议先获取单独提供的以下两个 patch，并分别在 `vLLM` 与 `vllm-ascend` 仓库执行：
>
> ```bash
> cd /vllm-workspace/vllm
> git am /workspace/kv_pool_patches/0001-Bugfix-Fix-negative-local_cache_hit-in-P-D-disaggreg.patch
>
> cd /vllm-workspace//vllm-ascend
> git am /workspace/kv_pool_patches/0001-BugFix-0.18.0-KV-Pool-Fix-KV-Pool-not-putting-kv-cac.patch
> ```
>
> 其中，`0001-Bugfix-Fix-negative-local_cache_hit-in-P-D-disaggreg.patch` 需要打到 `/vllm-workspace/vllm` 仓库下，用于修复 `local_cache_hit` 指标出现负值的问题；`0001-BugFix-0.18.0-KV-Pool-Fix-KV-Pool-not-putting-kv-cac.patch` 需要打到 `vllm-ascend` 仓库下，用于修复 vLLM v0.18.0 在 speculative decoding 场景下 KV Pool 未正确执行 KV Cache put / finalize 的问题，并规避后续 vLLM metrics 统计相关报错。若环境中已包含这些 patch 改动，可跳过此步骤。
>
> **当前方案说明**：Mooncake 后端已可通过 `AscendStoreConnector` 直接使用，推荐在 `kv-transfer-config`
> 中显式配置 `"backend": "mooncake"` 和 `"lookup_rpc_port"`。

## 环境准备

### 硬件要求

- **A3 单机**：1 × Atlas 800I A3 或 Atlas 800T A3 服务器，配备 8 张 NPU 卡（共 16 个 NPU device）
- **A2 双机**：2 × Atlas 800I A2 服务器，每台配备 8 张 NPU 卡
- 已配置 RoCE 或灵衢网络以获得最佳性能

### 软件要求

采用模型配套的 Docker 镜像，软件版本与 Docker 镜像内置版本保持一致，确保 HDK、固件等软件在配套范围内。

Mooncake 在 Ascend 场景下建议关注以下版本和环境变量：

- **A3 推荐方案**：当 `HDK >= 26.0.0` 且 `CANN >= 9.0.0` 时，推荐设置 `ASCEND_ENABLE_USE_FABRIC_MEM=1`
- **A3 兼容方案**：当 `25.5.0 <= HDK < 26.0.0` 时，可使用 `ASCEND_BUFFER_POOL=4:8`
- **A2 双机场景**：建议设置 `HCCL_INTRA_ROCE_ENABLE=1` 以支持跨机直传

### 模型权重

下载 GLM-5 W4A8 模型权重并放置到指定目录，如 `/home/models/GLM-5-w4a8/`。

> **模型下载地址**：[魔搭社区](https://modelscope.cn/models/Eco-Tech/GLM-5-w4a8)

## 使用 Docker 运行

本教程使用的 Docker 镜像版本为 `vllm-ascend:0.18.0rc1`。如本地尚未下载，可先按机器类型执行以下命令：

```bash
# A2 双机
docker pull quay.io/ascend/vllm-ascend:0.18.0rc1

# A3 单机
docker pull quay.io/ascend/vllm-ascend:0.18.0rc1-a3
```

如果下载较慢，可将 `quay.io` 替换为 `m.daocloud.io/quay.io` 或 `quay.nju.edu.cn` 以加速拉取。更多镜像说明可参考[安装文档](../../installation.md#set-up-using-docker)。

### A3 单机

可参考以下脚本启动 A3 容器。先将下面内容保存为 `start-docker.sh`：

```bash
#!/bin/bash
IMAGES_ID="$1"
NAME="$2"

# 检查参数数量（需要 2 个：镜像 ID 和容器名）
if [ $# -ne 2 ]; then
    echo "error: 需要传入2个参数，格式：$0 <镜像ID> <容器名>"
    exit 1
fi

# 检查镜像是否存在
if ! docker images --format "{{.ID}}" | grep -q "^${IMAGES_ID:0:12}$"; then
    echo "error: 镜像ID $IMAGES_ID 不存在"
    exit 1
fi

docker run --name "${NAME}" -it -d --net=host --shm-size=500g \
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

拉取镜像后，可先通过以下命令查看镜像 ID：

```bash
docker images | grep vllm-ascend
```

然后执行以下命令创建容器：

```bash
bash start-docker.sh ac1c767e5aa2 glm5-v0.18.0rc1-a3
```

创建完成后进入容器：

```bash
docker exec -it glm5-v0.18.0rc1-a3 bash
```

### A2 双机

A2 双机场景也可采用与 A3 类似的脚本方式。在两个节点分别保存同一份 `start-docker.sh`，内容如下：

```bash
#!/bin/bash
IMAGES_ID="$1"
NAME="$2"

# 检查参数数量（需要 2 个：镜像 ID 和容器名）
if [ $# -ne 2 ]; then
    echo "error: 需要传入2个参数，格式：$0 <镜像ID> <容器名>"
    exit 1
fi

# 检查镜像是否存在
if ! docker images --format "{{.ID}}" | grep -q "^${IMAGES_ID:0:12}$"; then
    echo "error: 镜像ID $IMAGES_ID 不存在"
    exit 1
fi

docker run --name "${NAME}" -it -d --net=host --shm-size=500g \
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

两个节点都先执行以下命令查看各自本机的镜像 ID：

```bash
docker images | grep vllm-ascend
```

然后分别创建容器，例如：

```bash
# 节点 0
bash start-docker.sh ac1c767e5aa2 glm5-v0.18.0rc1-a2-node0

# 节点 1
bash start-docker.sh ac1c767e5aa2 glm5-v0.18.0rc1-a2-node1
```

创建完成后分别进入容器：

```bash
# 节点 0
docker exec -it glm5-v0.18.0rc1-a2-node0 bash

# 节点 1
docker exec -it glm5-v0.18.0rc1-a2-node1 bash
```

## 准备 Mooncake

### 验证是否已安装

如果当前镜像或环境已经包含 Mooncake，可直接验证：

```bash
python -c "import mooncake; print(mooncake.__file__)"
```

若能正常输出安装路径，可跳过下文安装步骤。

### 安装 Mooncake

如果环境中未安装 Mooncake，可参考以下步骤编译安装：

```bash
git clone -b v0.3.9 --depth 1 https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
git submodule update --init --recursive
apt-get install mpich libmpich-dev -y
bash dependencies.sh -y
mkdir build
cd build
cmake .. -DUSE_ASCEND_DIRECT=ON
make -j
make install
```

验证安装：

```bash
python -c "import mooncake; print(mooncake.__file__)"
```

如果运行时报找不到 Mooncake 动态库，可按实际安装路径补充 `LD_LIBRARY_PATH`，例如：

```bash
export LD_LIBRARY_PATH=/usr/local/lib64/python3.11/site-packages/mooncake:$LD_LIBRARY_PATH
```

## 启动 Mooncake Master

选择一个节点或容器启动 `mooncake_master`。A3 单机场景启动一次即可；A2 双机场景建议在节点 0 启动，两个节点共用同一个 Master。

```bash
mooncake_master \
  --port 50088 \
  --eviction_high_watermark_ratio 0.9 \
  --eviction_ratio 0.1 \
  --default_kv_lease_ttl 11000
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `port` | `50088` | Master 服务端口 |
| `eviction_high_watermark_ratio` | `0.9` | 存储达到 90% 水位时触发淘汰 |
| `eviction_ratio` | `0.1` | 每次淘汰 10% 数据 |
| `default_kv_lease_ttl` | `11000` | KV 对象默认 TTL，建议大于 `ASCEND_CONNECT_TIMEOUT` 和 `ASCEND_TRANSFER_TIMEOUT` |

## 配置 mooncake.json

在每个参与部署的节点或容器中创建同一份 `mooncake.json`，例如放在 `/workspace/mooncake.json`：

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "<Mooncake Master 所在节点IP>:50088",
  "global_segment_size": "100GB"
}
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `metadata_server` | 固定为 `P2PHANDSHAKE` |
| `protocol` | Ascend 场景固定为 `ascend` |
| `device_name` | 保持空字符串即可 |
| `master_server_address` | Mooncake Master 的 IP 和端口 |
| `global_segment_size` | 每张卡向 KV Pool 注册的容量，建议按 1GB 对齐 |

### 导出配置环境变量

```bash
export MOONCAKE_CONFIG_PATH=/workspace/mooncake.json
```

## 部署 GLM-5 W4A8 与 Mooncake KV Pool

### A3 单机部署

创建启动脚本 `run_glm5_w4a8_mooncake_a3.sh`：

```bash
#!/bin/bash

# NPU 性能优化配置
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

# vLLM 配置
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export PYTHONHASHSEED=0

# Mooncake 配置
export MOONCAKE_CONFIG_PATH="/workspace/mooncake.json"
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

# A3 推荐配置：HDK >= 26.0.0 且 CANN >= 9.0.0 时启用
export ASCEND_ENABLE_USE_FABRIC_MEM=1
# 若不满足上述版本条件，可改用：
# export ASCEND_BUFFER_POOL=4:8

MODEL_PATH="/home/models/GLM-5-w4a8"

vllm serve $MODEL_PATH \
  --host 0.0.0.0 \
  --port 1025 \
  --data-parallel-size 1 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name glm-5 \
  --max-num-seqs 8 \
  --max-model-len 200000 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --quantization ascend \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --additional-config '{"multistream_overlap_shared_expert":true}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --kv-transfer-config '{
      "kv_connector": "AscendStoreConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
          "lookup_rpc_port": "0",
          "backend": "mooncake",
          "use_layerwise": false,
          "load_async": true
      }
  }' 2>&1 | tee ./glm-5_mooncake.log
```

运行脚本：

```bash
bash run_glm5_w4a8_mooncake_a3.sh
```

### A2 双机部署

A2 双机场景下，两个节点应共用同一个 `mooncake_master` 和同一份 `mooncake.json`，但 `lookup_rpc_port` 必须区分。

**节点 0** 创建 `run_glm5_w4a8_mooncake_a2_node0.sh`：

```bash
#!/bin/bash

# 网络配置 - 根据实际环境修改
nic_name="bond0"
local_ip="100.100.135.173"
node0_ip="100.100.135.173"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

# NPU 性能优化配置
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# vLLM 配置
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export PYTHONHASHSEED=0

# Mooncake 配置
export MOONCAKE_CONFIG_PATH="/workspace/mooncake.json"
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_BUFFER_POOL=4:8

MODEL_PATH="/home/models/GLM-5-w4a8"

vllm serve $MODEL_PATH \
  --host 0.0.0.0 \
  --port 1025 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address $node0_ip \
  --data-parallel-rpc-port 12890 \
  --tensor-parallel-size 8 \
  --quantization ascend \
  --seed 1024 \
  --served-model-name glm-5 \
  --enable-expert-parallel \
  --max-num-seqs 2 \
  --max-model-len 131072 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"multistream_overlap_shared_expert":true}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --kv-transfer-config '{
      "kv_connector": "AscendStoreConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
          "lookup_rpc_port": "0",
          "backend": "mooncake",
          "use_layerwise": false,
          "load_async": true
      }
  }' 2>&1 | tee ./glm-5_mooncake.log
```

**节点 1** 创建 `run_glm5_w4a8_mooncake_a2_node1.sh`：

```bash
#!/bin/bash

# 网络配置 - 根据实际环境修改
nic_name="enp61s0f2"
local_ip="100.100.135.190"
node0_ip="100.100.135.173"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

# NPU 性能优化配置
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# vLLM 配置
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export PYTHONHASHSEED=0

# Mooncake 配置
export MOONCAKE_CONFIG_PATH="/workspace/mooncake.json"
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_BUFFER_POOL=4:8

MODEL_PATH="/home/models/GLM-5-w4a8"

vllm serve $MODEL_PATH \
  --host 0.0.0.0 \
  --port 1026 \
  --headless \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address $node0_ip \
  --data-parallel-rpc-port 12890 \
  --tensor-parallel-size 8 \
  --quantization ascend \
  --seed 1024 \
  --served-model-name glm-5 \
  --enable-expert-parallel \
  --max-num-seqs 2 \
  --max-model-len 131072 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"multistream_overlap_shared_expert":true}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --kv-transfer-config '{
      "kv_connector": "AscendStoreConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
          "lookup_rpc_port": "1",
          "backend": "mooncake",
          "use_layerwise": false,
          "load_async": true
      }
  }' 2>&1 | tee -a ./glm-5_mooncake.log
```

运行脚本：

```bash
# 节点 0 先启动
bash run_glm5_w4a8_mooncake_a2_node0.sh

# 节点 1 后启动
bash run_glm5_w4a8_mooncake_a2_node1.sh
```

### 配置参数说明

#### A3 单机参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `tensor-parallel-size` | `16` | 使用 16 个 NPU device |
| `data-parallel-size` | `1` | 数据并行大小 |
| `max-model-len` | `200000` | 最大上下文长度 |
| `max-num-batched-tokens` | `4096` | 最大批处理 token 数 |
| `max-num-seqs` | `8` | 最大并发序列数 |
| `gpu-memory-utilization` | `0.95` | NPU 显存利用率 |
| `quantization` | `ascend` | 使用 Ascend 量化 |
| `kv_connector` | `AscendStoreConnector` | 统一的 KV Pool 连接器 |
| `kv_role` | `kv_both` | 同时支持读写外部 KV Cache |
| `backend` | `mooncake` | 使用 Mooncake 作为 KV Pool 后端 |
| `lookup_rpc_port` | `0` | 本机查找服务端口编号 |

#### A2 双机参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `tensor-parallel-size` | `8` | 每节点使用 8 张 NPU 卡 |
| `data-parallel-size` | `2` | 数据并行大小（2 节点） |
| `data-parallel-size-local` | `1` | 本地数据并行大小 |
| `max-model-len` | `131072` | 最大上下文长度 |
| `max-num-batched-tokens` | `4096` | 最大批处理 token 数 |
| `max-num-seqs` | `2` | 最大并发序列数 |
| `lookup_rpc_port (node0)` | `0` | 节点 0 的查找服务端口编号 |
| `lookup_rpc_port (node1)` | `1` | 节点 1 的查找服务端口编号 |

### 环境变量说明

| 环境变量 | 值 | 说明 |
|----------|-----|------|
| `MOONCAKE_CONFIG_PATH` | `/workspace/mooncake.json` | Mooncake 配置文件绝对路径 |
| `HCCL_RDMA_TIMEOUT` | `17` | RDMA 最小重传超时参数 |
| `ASCEND_CONNECT_TIMEOUT` | `10000` | 直传连接建立超时，单位毫秒 |
| `ASCEND_TRANSFER_TIMEOUT` | `10000` | 直传数据传输超时，单位毫秒 |
| `ASCEND_ENABLE_USE_FABRIC_MEM` | `1` | A3 推荐配置，启用统一地址直传 |
| `ASCEND_BUFFER_POOL` | `4:8` | 兼容方案，配置 NPU 侧 buffer pool |
| `HCCL_INTRA_ROCE_ENABLE` | `1` | A2 双机场景建议开启 |
| `PYTHONHASHSEED` | `0` | 保证 KV Cache 键计算一致 |
| `VLLM_USE_V1` | `1` | 启用 vLLM v1 架构 |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `1800` | 引擎就绪超时时间 |

## 功能验证

服务启动后，验证部署是否成功。

### 测试推理

```bash
curl -H "Accept: application/json" \
    -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "glm-5",
        "messages": [{
            "role": "user",
            "content": "你好，请介绍一下人工智能的未来发展趋势。"
        }],
        "stream": false,
        "ignore_eos": false,
        "temperature": 0,
        "max_tokens": 200
    }' http://localhost:1025/v1/chat/completions
```

## 缓存命中率监控

### 查看 vLLM 日志

日志文件命名格式：`glm-5_mooncake.log`

```bash
# 查看最新日志
tail -f glm-5_mooncake.log

# 实时监控关键日志
tail -f glm-5_mooncake.log | grep -E "mooncake|external_prefix_cache|cache|hit_rate"
```

### 查看 Prefix Cache 命中率

```bash
# 通过日志查看
grep -E "prefix cache hit|cache hit rate" glm-5_mooncake.log

# 通过 metrics API 查看
curl http://localhost:1025/metrics | grep -E "vllm_prefix_cache|cache_hit"
```

### 查看 Mooncake 外部 KV Cache 命中率

```bash
# 查看 Mooncake 相关日志
grep -E "mooncake|Mooncake|external_prefix_cache|kvpool" glm-5_mooncake.log

# 查看外部缓存指标
curl http://localhost:1025/metrics | grep external_prefix_cache
```

### 使用脚本持续监控命中率

如果当前环境包含 `vllm-ascend` 仓库源码，也可以直接使用仓库自带脚本持续观测命中率：

```bash
# 在 vllm-ascend 仓库根目录执行
bash tools/watch_cache_hit_rate.sh -u http://localhost:1025/metrics -i 10

# 如需同时保存到文件
bash tools/watch_cache_hit_rate.sh \
  -u http://localhost:1025/metrics \
  -i 10 \
  -o cache_hit_rate.log
```

脚本会同时输出每个 engine 以及汇总行 `all` 的命中率，常用字段如下：

- `local_win`：vLLM 本地 Prefix Cache 的窗口命中率
- `local_total`：vLLM 本地 Prefix Cache 的累计命中率
- `ext_win`：Mooncake 外部 KV Cache 的窗口命中率
- `ext_total`：Mooncake 外部 KV Cache 的累计命中率
- `eff_total`：综合本地和外部缓存后的端到端有效命中率

### 指标说明

| 指标 | 说明 | 统计方式 |
|------|------|----------|
| Prefix cache hit rate | 本地 HBM 命中率 | 最近 1000 个请求的滑动窗口 |
| External prefix cache hit rate | Mooncake 外部 KV Cache 命中率 | 从服务启动到当前时刻的累计统计 |
| TTFT | 首 token 延迟 | 单次请求指标 |

## 性能测试

### 选项一：快速测试

使用 `vllm bench serve` 对已启动的 OpenAI 兼容服务进行快速压测。下面示例默认以本机服务为目标：

```bash
BENCH_HOST=127.0.0.1
BENCH_PORT=1025
TOKENIZER_PATH=/home/models/GLM-5-w4a8

vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 31744 \
  --prefix-repetition-suffix-len 1024 \
  --prefix-repetition-output-len 2048 \
  --num-prompts 100 \
  --prefix-repetition-num-prefixes 5 \
  --ignore-eos \
  --model glm-5 \
  --tokenizer $TOKENIZER_PATH \
  --seed 1000 \
  --host $BENCH_HOST \
  --port $BENCH_PORT \
  --max-concurrency 10
```

参数含义如下：

| 参数 | 含义 |
|------|------|
| `--backend openai-chat` | 按 OpenAI Chat Completions 接口格式发请求 |
| `--endpoint /v1/chat/completions` | 指定压测目标接口路径 |
| `--dataset-name prefix_repetition` | 使用前缀重复数据集，便于观察缓存收益 |
| `--prefix-repetition-prefix-len 31744` | 共享前缀长度 |
| `--prefix-repetition-suffix-len 1024` | 每条请求的独立后缀长度 |
| `--prefix-repetition-output-len 2048` | 生成输出长度 |
| `--num-prompts 100` | 总请求数 |
| `--prefix-repetition-num-prefixes 5` | 共享前缀模板数量 |
| `--ignore-eos` | 忽略 EOS，尽量生成到目标长度 |
| `--model glm-5` | 请求时使用的模型名，需要与服务端 `served-model-name` 一致 |
| `--tokenizer $TOKENIZER_PATH` | tokenizer 路径，需指向本地 GLM-5 W4A8 模型目录 |
| `--seed 1000` | 固定随机种子，便于复现 |
| `--host $BENCH_HOST` | 压测目标地址 |
| `--port $BENCH_PORT` | 压测目标端口；若服务不是 `1025`，请改成实际端口 |
| `--max-concurrency 10` | 最大并发请求数 |

### 选项二：深度测试

使用 AISBench 进行深度性能测试：

```bash
ais_bench --models vllm_api_stream_chat \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --debug --summarizer default_perf --mode perf
```

## 故障排除

### 常见问题

1. **`MOONCAKE_CONFIG_PATH` 未设置**

   检查环境变量和配置文件路径：
   ```bash
   echo $MOONCAKE_CONFIG_PATH
   cat $MOONCAKE_CONFIG_PATH
   ```

2. **Mooncake Master 无法连接**

   确认 `master_server_address` 可达：
   ```bash
   grep master_server_address /workspace/mooncake.json
   netstat -tlnp | grep 50088
   ```

3. **`import mooncake` 失败**

   重新验证安装：
   ```bash
   python -c "import mooncake; print(mooncake.__file__)"
   ```

4. **动态库加载失败**

   按实际路径补充 `LD_LIBRARY_PATH`：
   ```bash
   export LD_LIBRARY_PATH=/usr/local/lib64/python3.11/site-packages/mooncake:$LD_LIBRARY_PATH
   ```

5. **KV Cache 未命中**

   验证 `PYTHONHASHSEED` 与模型路径保持一致：
   ```bash
   echo $PYTHONHASHSEED
   ```

6. **A2 双机跨节点通信失败**

   检查网卡、IP 和 RoCE 配置：
   ```bash
   ifconfig
   echo $HCCL_INTRA_ROCE_ENABLE
   ```

7. **同机多实例查找端口冲突**

   为每个实例设置不同的 `lookup_rpc_port`：
   ```bash
   # 例如同机两个实例分别使用 0 和 1
   ```

8. **引擎启动超时**

   增加 `VLLM_ENGINE_READY_TIMEOUT_S`：
   ```bash
   export VLLM_ENGINE_READY_TIMEOUT_S=3600
   ```

9. **性能测试首轮偏慢**

   Mooncake 在开启直传和外部 KV Cache 后，首次请求可能包含额外的连接建立开销。建议在正式压测前先做一轮 warm-up。

### 日志查看

```bash
tail -f glm-5_mooncake.log
```

## 参考资料

- [Mooncake 项目](https://github.com/kvcache-ai/Mooncake)
- [vLLM Ascend KV Pool 说明](../../user_guide/feature_guide/kv_pool.md)
- [vLLM Ascend 文档](https://docs.vllm.ai/projects/ascend/)
- [GLM-5 教程](../../tutorials/models/GLM5.md)
