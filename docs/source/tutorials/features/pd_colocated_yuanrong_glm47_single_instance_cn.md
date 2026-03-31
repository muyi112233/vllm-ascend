# 基于 Yuanrong 的 GLM-4.7-Flash 单实例部署

## 概述

本指南提供在单台 Atlas 800T/I A2 服务器（4 张 NPU 卡）上部署 GLM-4.7-Flash 模型并使用
Yuanrong Datasystem 作为 KV Pool 后端的详细步骤。

GLM-4.7-Flash 是采用混合专家架构的高效推理模型，专为智能体应用设计。使用 Yuanrong 作为
KV Pool 后端可以实现高效的 KV Cache 存储和请求间的复用。

> **前置条件**：本教程使用的 Docker 镜像需要线下单独获取，请联系相关人员获取镜像版本号。

## 环境准备

### 硬件要求

- 1 × Atlas 800T/I A2 服务器，配备 4 张 NPU 卡（每张 64G 显存）
- 已配置 RoCE 网络以获得最佳性能

### 软件要求

软件版本与 Docker 镜像内置版本保持一致。

### 模型权重

下载 GLM-4.7-Flash 模型权重并放置到指定目录，如 `/home/models/GLM-4.7-Flash/`。

> **模型下载地址**：[魔搭社区](https://modelscope.cn/models/ZhipuAI/GLM-4.7-Flash) | [Hugging Face](https://huggingface.co/zai-org/GLM-4.7-Flash)

## 使用 Docker 运行

启动包含 4 张 NPU 卡的 Docker 容器：

> **注意**：Docker 镜像需要线下单独获取，示例镜像名为 `cwb_glm47_flash_vllm_patch_new_cann_litellm:v1`。

```bash
export IMAGE=cwb_glm47_flash_vllm_patch_new_cann_litellm:v1
export NAME=vllm-ascend

docker run --rm \
--name $NAME \
--shm-size=1g \
--net=host \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

## 安装 Yuanrong Datasystem

### 在线安装

```bash
pip install openyuanrong-datasystem
```

### 离线安装

如果目标环境无外网权限，请联系相关人员获取 `openyuanrong-datasystem` 及其依赖的 whl 包，然后进行离线安装：

```bash
# 将获取的 whl 包放置到 /mnt/packages 目录后安装
pip install --no-index --find-links=/mnt/packages openyuanrong-datasystem
```

验证安装：

```bash
python -c "import yr.datasystem; print('Yuanrong Datasystem 安装成功')"
```

## 启动 Yuanrong 服务

本教程为单实例部署方式，etcd 和 Datasystem Worker 均在 vLLM-Ascend 容器内启动。

创建启动脚本 `run_yr.sh`：

```bash
#!/bin/bash

# 配置参数
export HOST_IP="<您的节点IP地址>"
export ETCD_IP="${HOST_IP}"
export WORKER_PORT=31501
export ETCD_PORT=2379
export SHM_SIZE=512000
export NODE_TIMEOUT=600
export NODE_DEAD_TIMEOUT=600000
export LIVENESS_PATH=/workspace/liveness

# 启动 etcd（单实例模式）
etcd \
  --name etcd-single \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://${ETCD_IP}:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://${ETCD_IP}:2380 \
  --initial-cluster etcd-single=http://${ETCD_IP}:2380 \
  > /tmp/etcd.log 2>&1 &

# 等待 etcd 启动
sleep 3

# 验证 etcd 是否正常运行
etcdctl --endpoints "${ETCD_IP}:2379" put key "value"
etcdctl --endpoints "${ETCD_IP}:2379" get key

# 启动 Datasystem Worker
dscli start -w \
  --worker_address ${HOST_IP}:${WORKER_PORT} \
  --etcd_address ${ETCD_IP}:${ETCD_PORT} \
  --shared_memory_size_mb ${SHM_SIZE} \
  --node_timeout_s ${NODE_TIMEOUT} \
  --node_dead_timeout_s ${NODE_DEAD_TIMEOUT} \
  --liveness_check_path ${LIVENESS_PATH}

echo "Yuanrong 服务启动完成"
echo "etcd 日志: /tmp/etcd.log"
```

运行脚本：

```bash
bash run_yr.sh
```

### etcd 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| name | etcd-single | etcd 节点名称，集群中必须唯一 |
| data-dir | /tmp/etcd-data | 数据存储目录，用于持久化保存 etcd 数据 |
| listen-client-urls | http://0.0.0.0:2379 | 监听客户端请求的 URL 地址，`0.0.0.0` 表示监听所有网卡 |
| advertise-client-urls | http://${ETCD_IP}:2379 | 对外广播的客户端 URL，其他节点通过此地址连接 |
| listen-peer-urls | http://0.0.0.0:2380 | 监听集群节点间通信的 URL 地址 |
| initial-advertise-peer-urls | http://${ETCD_IP}:2380 | 对外广播的集群通信 URL，其他 etcd 节点通过此地址进行数据同步 |
| initial-cluster | etcd-single=http://${ETCD_IP}:2380 | 初始集群配置，格式为 `节点名=URL`，多节点时用逗号分隔 |

> **参考文档**：[etcd 官方文档](https://etcd.io/docs/)，了解更多参数配置。
>
> **生产环境建议**：上述示例为单实例部署，适用于测试和开发环境。对于可靠性要求较高的生产环境，建议部署 etcd 集群（通常 3 或 5 个节点）。集群部署请参考 [etcd 集群部署指南](https://etcd.io/docs/latest/op-guide/clustering/)。

### Datasystem Worker 参数说明

| 参数                   | 值                           | 说明                           |
| ---------------------- | ---------------------------- | ------------------------------ |
| worker_address         | ${HOST_IP}:${WORKER_PORT}    | Worker 服务地址和端口          |
| etcd_address           | ${ETCD_IP}:${ETCD_PORT}      | etcd 服务发现地址              |
| shared_memory_size_mb  | ${SHM_SIZE} (512000)        | 共享内存大小（500 GB）         |
| node_timeout_s         | ${NODE_TIMEOUT} (600)        | 节点超时时间（秒）             |
| node_dead_timeout_s    | ${NODE_DEAD_TIMEOUT} (600000)| 节点死亡超时时间（毫秒）       |
| liveness_check_path    | ${LIVENESS_PATH}             | 存活检查路径                   |

> **参考文档**：[Yuanrong Datasystem 文档](https://atomgit.com/openeuler/yuanrong-datasystem)，了解更多 worker 参数和环境变量配置。

停止 Worker：

```bash
dscli stop --worker_address ${HOST_IP}:${WORKER_PORT}
```

## 部署 GLM-4.7-Flash 与 Yuanrong KV Pool

### 启动脚本

创建启动脚本 `run_glm47_flash_yuanrong.sh`：

```bash
#!/bin/bash

LOG_FILE="vllm_glm47_yuanrong_$(date +%Y%m%d_%H%M%S).log"

cat "$0" > ${LOG_FILE}

# vLLM v1 架构配置
export VLLM_USE_V1=1
export CPU_AFFINITY_CONF=
export TASK_QUEUE_ENABLE=1

# NPU 性能优化配置
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 内存分配器优化
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_LIBRARY_PATH

# Yuanrong Datasystem 配置
export PYTHONHASHSEED=0
# 必填：必须与本地 dscli --worker_address 一致
export DS_WORKER_ADDR="${HOST_IP}:${WORKER_PORT}"
# 可选（默认值：0）
export DS_ENABLE_EXCLUSIVE_CONNECTION=0
export DS_ENABLE_REMOTE_H2D=0

vllm serve /home/models/GLM-4.7-Flash \
  --served-model-name GLM-4.7-Flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  --max-model-len 202752 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 20 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --host 0.0.0.0 \
  --port 1024 \
  --gpu-memory-utilization 0.92 \
  --enable-expert-parallel \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --additional-config '{"enable_cpu_binding":true}' \
  --compilation-config '{"cudagraph_capture_sizes": [1,2,4,8,16,32,64,128,256,512], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --kv-transfer-config '{
      "kv_connector": "AscendStoreConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
          "lookup_rpc_port": "0",
          "backend": "yuanrong",
          "load_async": true
      }
  }' 2>&1 | tee -a ${LOG_FILE}
```

### 配置参数说明

| 参数                     | 值                      | 说明                           |
| ------------------------ | ----------------------- | ------------------------------ |
| tensor-parallel-size     | 4                       | 使用 4 张 NPU 卡               |
| data-parallel-size       | 1                       | 数据并行大小                   |
| max-model-len            | 202752                  | 最大上下文长度（根据实际需求调整）|
| max-num-batched-tokens   | 8192                    | 最大批处理 token 数（分块预填充，根据性能调整）|
| max-num-seqs             | 20                      | 最大并发序列数                 |
| gpu-memory-utilization   | 0.92                    | GPU 显存利用率                 |
| enable-expert-parallel   | (标志)                  | 启用 MoE 专家并行              |
| enable-chunked-prefill   | (标志)                  | 启用分块预填充                 |
| enable-auto-tool-choice  | (标志)                  | 启用自动工具选择               |
| tool-call-parser         | glm47                   | 工具调用解析器                 |
| compilation-config       | cudagraph 配置          | CudaGraph 捕获配置（加速解码） |
| kv_connector             | AscendStoreConnector    | 使用 AscendStoreConnector      |
| kv_role                  | kv_both                 | 同时支持生产和消费             |
| backend                  | yuanrong                | 使用 Yuanrong 后端             |
| load_async               | true                    | 启用异步加载                   |

### 环境变量说明

| 环境变量 | 值 | 说明 |
|----------|-----|------|
| `VLLM_USE_V1` | 1 | 启用 vLLM v1 架构（新版调度器） |
| `CPU_AFFINITY_CONF` | (空) | CPU 亲和性配置 |
| `TASK_QUEUE_ENABLE` | 1 | 启用任务队列 |
| `HCCL_OP_EXPANSION_MODE` | AIV | HCCL 算子扩展模式（AI Vector 优化） |
| `PYTORCH_NPU_ALLOC_CONF` | expandable_segments:True | NPU 显存分配策略（减少碎片） |
| `LD_LIBRARY_PATH` | jemalloc 路径 | 高性能内存分配器 |
| `PYTHONHASHSEED` | 0 | Python 哈希种子，确保 KV Cache 键一致性 |
| `DS_WORKER_ADDR` | ${HOST_IP}:${WORKER_PORT} | Yuanrong Worker 地址，必须与 dscli 启动参数一致 |
| `DS_ENABLE_EXCLUSIVE_CONNECTION` | 0 | 是否启用独占连接（可选，默认 0） |
| `DS_ENABLE_REMOTE_H2D` | 0 | 是否启用远程 Host-to-Device 传输（可选，默认 0） |

## 功能验证

服务启动后，验证部署是否成功：

### 检查服务状态

```bash
curl http://localhost:1024/health
```

### 测试推理

```bash
curl -H "Accept: application/json" \
    -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "GLM-4.7-Flash", 
        "messages": [{ 
            "role": "user", 
            "content": "你好，请介绍一下人工智能的未来发展趋势。" 
        }], 
        "stream": false, 
        "ignore_eos": false, 
        "temperature": 0, 
        "max_tokens": 200 
    }' http://localhost:1024/v1/chat/completions
```

### 测试 KV Cache 复用

发送相同的提示词两次，验证 KV Cache 是否被复用：

```bash
# 第一次请求 - 填充 KV Cache
curl -H "Accept: application/json" \
    -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "GLM-4.7-Flash", 
        "messages": [{ 
            "role": "user", 
            "content": "请详细解释机器学习的概念和基本原理。" 
        }], 
        "stream": false, 
        "temperature": 0, 
        "max_tokens": 100 
    }' http://localhost:1024/v1/chat/completions

# 第二次请求 - 命中 KV Cache（TTFT 更快）
curl -H "Accept: application/json" \
    -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "GLM-4.7-Flash", 
        "messages": [{ 
            "role": "user", 
            "content": "请详细解释机器学习的概念和基本原理。" 
        }], 
        "stream": false, 
        "temperature": 0, 
        "max_tokens": 100 
    }' http://localhost:1024/v1/chat/completions
```

## 缓存命中率监控

### 查看 vLLM 日志

日志文件命名格式：`vllm_glm47_yuanrong_YYYYMMDD_HHMMSS.log`

```bash
# 查看最新日志
tail -f vllm_glm47_yuanrong_*.log

# 实时监控关键指标
tail -f vllm_glm47_yuanrong_*.log | grep -E "hit_rate|cache|KV"
```

### 查看 Prefix Cache 命中率

vLLM 内置的 prefix cache 命中率统计：

```bash
# 方法一：通过日志查看
grep -E "prefix cache hit|cache hit rate" vllm_glm47_yuanrong_*.log

# 方法二：通过 metrics API 查看（如果启用了）
curl http://localhost:1024/metrics | grep -E "vllm_prefix_cache|cache_hit"
```

### 查看 Yuanrong KV Cache 命中率

Yuanrong 后端的缓存命中率通过日志输出：

```bash
# 查看 Yuanrong 相关日志
grep -E "yuanrong|YuanrongBackend|kv_cache|hit_rate" vllm_glm47_yuanrong_*.log

# 查看详细的缓存操作日志
grep -E "get|put|exists|HeteroClient" vllm_glm47_yuanrong_*.log
```

### 缓存命中率指标说明

| 指标 | 说明 | 统计方式 |
|------|------|----------|
| Prefix Cache Hit Rate | **HBM（本地显存）**命中率 | **滑动窗口**：最近 1000 个请求 |
| External Cache Hit Rate | **Yuanrong（外部 KV Cache）**命中率 | **累计统计**：从服务启动到当前时刻 |
| TTFT (Time to First Token) | 首个 token 延迟 | 单次请求指标，命中率高时 TTFT 显著降低 |

**统计周期说明**：
- **HBM 命中率**：来自 vLLM 上游 `CachingMetrics` 模块，使用滑动窗口统计最近 1000 个请求
- **Yuanrong 命中率**：来自 Prometheus Counter 指标，**累计统计**从服务启动到当前时刻的总命中数和查询数
- 日志每 **10 秒** 输出一次 HBM 命中率
- HBM 命中率计算公式：`hit_rate = window_hits / window_queries`
- Yuanrong 命中率计算公式：`hit_rate = total_hits / total_queries`

**查看 Yuanrong 外部缓存命中率**：
```bash
# 通过 Prometheus metrics 查看外部缓存统计
curl http://localhost:1024/metrics | grep external_prefix_cache

# 输出示例：
# vllm:external_prefix_cache_queries_total{...} 10000
# vllm:external_prefix_cache_hits_total{...} 8500
# 命中率 = hits / queries = 85%
```

**注意**：外部缓存指标为 Prometheus Counter 类型，表示从服务启动以来的累计值。

**如何查看单次请求是否命中**：
```bash
# 查看单个请求的缓存命中情况
grep "num_computed_tokens" vllm_glm47_yuanrong_*.log

# 如果 num_computed_tokens > 0，表示该请求命中了缓存
# num_computed_tokens 表示从缓存中复用的 token 数量
```

### 使用 Prometheus 监控（可选）

如果需要持续监控，可以启用 vLLM 的 Prometheus metrics：

```bash
# 启动时添加 metrics 端口
vllm serve ... --enable-metrics --metrics-port 8001
```

然后通过 Prometheus 抓取 `http://localhost:8001/metrics`。

## 性能测试

### 选项一：快速测试

使用 vLLM Benchmark 进行快速性能测试：

```shell
vllm bench serve \
  --backend vllm \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 8192 \
  --prefix-repetition-suffix-len 1024 \
  --prefix-repetition-output-len 256 \
  --num-prompts 10 \
  --prefix-repetition-num-prefixes 5 \
  --ignore-eos \
  --model GLM-4.7-Flash \
  --tokenizer /home/models/GLM-4.7-Flash \
  --seed 1000 \
  --host 0.0.0.0 \
  --port 1024 \
  --endpoint /v1/completions \
  --max-concurrency 4 \
  --request-rate 1
```

### 选项二：深度测试

使用 AISBench 进行深度性能测试：

```shell
ais_bench --models vllm_api_stream_chat \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --debug --summarizer default_perf --mode perf
```

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
   netstat -tlnp | grep 31501
   ```

3. **KV Cache 未找到**

   验证 `PYTHONHASHSEED` 设置一致：
   ```bash
   echo $PYTHONHASHSEED
   ```

4. **yr.datasystem 导入错误**

   确保 `openyuanrong-datasystem` 已安装：
   ```bash
   pip install openyuanrong-datasystem
   python -c "from yr.datasystem.hetero_client import HeteroClient; print('OK')"
   ```

5. **内存不足（OOM）**

   降低 `--max-num-seqs` 和 `--max-model-len`：
   ```bash
   --max-model-len 131072 \
   --max-num-seqs 10
   ```

6. **FIA 算子性能问题**

   执行 FIA 算子替换脚本：
   ```bash
   # A2 系列
   bash tools/install_flash_infer_attention_score_ops_a2.sh
   ```

### 日志查看

查看 vLLM 日志以获取详细错误信息：
```bash
tail -f vllm_glm47_yuanrong_*.log
```

## 参考资料

- [Yuanrong Datasystem 文档](https://atomgit.com/openeuler/yuanrong-datasystem)
- [etcd 文档](https://etcd.io/docs/)
- [vLLM Ascend 文档](https://docs.vllm.ai/projects/ascend/)
- [GLM-4.x 教程](../../tutorials/models/GLM4.x.md)
