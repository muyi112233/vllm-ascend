# `run_yr_worker_spill_ssd.sh` 使用说明

`docs/source/tutorials/features/run_yr_worker_spill_ssd.sh` 是一个单机测试脚本，用于启动 `etcd` 和 Yuanrong Datasystem Worker，并开启本地 SSD spill。它主要用于验证 `yuanrong_backend` 的本地 SSD 冷层行为。

这个脚本适合以下场景：

- 只想测试 `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py` 的本地 SSD spill
- 希望脚本自己启动单实例 `etcd`
- 需要一个和 GLM-5 Yuanrong 教程参数接近、但额外开启 SSD spill 的 Worker 启动模板

这个脚本不适合作为生产部署模板。它会清理本机旧的 `etcd`、`datasystem_worker` 和本地 datasystem 状态，目标是方便反复测试。

## 前置条件

- 已安装 `openyuanrong-datasystem`，并且 `dscli` 命令可用
- 当前目录下有可执行的 `./etcd`，或者通过 `ETCD_BIN` 指定 `etcd` 路径
- `etcdctl` 在 `PATH` 中，或者通过 `ETCDCTL_BIN` 指定路径
- 机器上有可写入的本地 SSD 路径，例如 `/data/ssd/yr_kv_spill`
- 当前仓库中已包含 `run_yr_worker_spill_ssd.sh` 脚本

## 快速使用

在仓库根目录执行：

```bash
chmod +x docs/source/tutorials/features/run_yr_worker_spill_ssd.sh

bash docs/source/tutorials/features/run_yr_worker_spill_ssd.sh \
  100.100.135.173 \
  18481 \
  19099
```

三个位置参数分别是：

```text
HOST_IP WORKER_PORT ETCD_PORT
```

如果不传参数，默认值是：

```text
HOST_IP=100.100.135.173
WORKER_PORT=18481
ETCD_PORT=19099
ETCD_PEER_PORT=19100
```

vLLM 侧连接这个 Worker 时，对应配置：

```bash
export DS_WORKER_ADDR=100.100.135.173:18481
```

## 脚本行为

脚本会执行以下动作：

- 取消 `http_proxy`、`https_proxy`、`HTTP_PROXY`、`HTTPS_PROXY`
- 尝试执行 `ulimit -l unlimited`
- 停止旧的 `etcd` 和 `datasystem_worker` 进程
- 删除 `/tmp/etcd-yuanrong`
- 删除当前目录下的 `./datasystem`
- 删除 `~/.datasystem`
- 创建 `SPILL_DIR`
- 启动单实例 `etcd`
- 等待 `etcd` health check 通过
- 使用 `dscli start -w` 启动 Worker，并开启 SSD spill

因此不要在有其他重要 `etcd` 或 datasystem 进程的机器上直接运行这个测试脚本。

## 参数说明

脚本支持位置参数和环境变量两种方式。位置参数优先级更高。

| 参数 | 默认值 | 含义 |
|---|---|---|
| 第 1 个参数 / `HOST_IP` | `100.100.135.173` | `etcd` 和 Worker 监听地址 |
| 第 2 个参数 / `WORKER_PORT` | `18481` | Worker 端口 |
| 第 3 个参数 / `ETCD_PORT` | `19099` | etcd client 端口 |
| `ETCD_PEER_PORT` | `ETCD_PORT + 1` | etcd peer 端口 |
| `ETCD_BIN` | `./etcd` | etcd 可执行文件路径 |
| `ETCDCTL_BIN` | `etcdctl` | etcdctl 可执行文件路径 |
| `ETCD_DATA_DIR` | `/tmp/etcd-yuanrong` | etcd 数据目录 |
| `SHM_SIZE_MB` | `512000` | Worker 共享内存大小 |
| `NODE_TIMEOUT` | `30` | 节点超时参数 |
| `NODE_DEAD_TIMEOUT` | `60` | 节点失活超时参数 |
| `LIVENESS_PATH` | `/workspace/liveness` | liveness 检查文件路径 |
| `ARENA_PER_TENANT` | `1` | `dscli start` 的 `arena_per_tenant` 参数 |
| `SPILL_DIR` | `/data/ssd/yr_kv_spill` | spill 根目录 |
| `SPILL_SIZE_LIMIT` | `214748364800` | spill 总容量上限，单位 Byte，默认约 200 GiB |
| `SPILL_THREAD_NUM` | `8` | spill 写盘并发度 |
| `SPILL_FILE_MAX_SIZE_MB` | `200` | 单个 spill 文件最大大小 |
| `SPILL_FILE_OPEN_LIMIT` | `512` | spill 允许同时打开的文件数上限 |
| `SPILL_ENABLE_READAHEAD` | `true` | 是否启用 spill 文件 readahead |
| `LOG_MONITOR` | `true` | 是否开启资源日志 |
| `LOG_MONITOR_EXPORTER` | `harddisk` | 资源日志 exporter |
| `LOG_MONITOR_INTERVAL_MS` | `5000` | 资源日志采集间隔 |

## 常见用法

### 1. 使用默认 spill 配置启动

```bash
bash docs/source/tutorials/features/run_yr_worker_spill_ssd.sh \
  100.100.135.173 \
  18481 \
  19099
```

### 2. 指定 SSD 目录和容量上限

```bash
export SPILL_DIR=/data/ssd/yr_kv_spill
export SPILL_SIZE_LIMIT=214748364800

bash docs/source/tutorials/features/run_yr_worker_spill_ssd.sh \
  100.100.135.173 \
  18481 \
  19099
```

### 3. 更容易触发 spill

默认 `SHM_SIZE_MB=512000`，和主教程里的大内存配置保持一致。如果你只是想快速验证 spill，可以把共享内存调小：

```bash
export SHM_SIZE_MB=4096
export SPILL_DIR=/data/ssd/yr_kv_spill
export SPILL_SIZE_LIMIT=53687091200

bash docs/source/tutorials/features/run_yr_worker_spill_ssd.sh \
  100.100.135.173 \
  18481 \
  19099
```

### 4. 调整 SSD 并发参数

```bash
export SPILL_THREAD_NUM=4
export SPILL_FILE_OPEN_LIMIT=256

bash docs/source/tutorials/features/run_yr_worker_spill_ssd.sh \
  100.100.135.173 \
  18481 \
  19099
```

## 如何验证 spill 已经发生

脚本只负责启动 `etcd` 和 Worker。要看到 SSD spill，仍然需要用 `yuanrong_backend` 或其他 client 持续写入数据，直到内存压力触发驱逐。

### 1. 看 spill 空间占用

默认 Worker 日志目录通常在 `~/datasystem/logs/worker/`。可以先看 `resource.log`：

```bash
tail -f ~/datasystem/logs/worker/resource.log
```

如果 spill 磁盘信息对应的使用量持续增长，说明数据已经开始占用本地 SSD。

### 2. 看单次 SSD 读盘耗时

当对象已经 spill 到 SSD 后，后续 `get` 命中该对象时，Worker 运行日志里会出现 `Read object [...] cost: ...ms` 这样的记录：

```bash
grep "Read object \[" ~/datasystem/logs/worker/*.INFO.log
```

这里看到的是单次从 spill 文件读取对象的耗时，适合确认这次读取是否真的走了 SSD 路径。

## 使用 `yuanrong_backend` 时的建议

- 当前 `yuanrong_backend` 默认使用 `WriteMode.NONE_L2_CACHE_EVICT`
- 这个脚本适合验证“内存 -> 本地 SSD spill -> get 时回填”的流程
- 如果只是做 `exist` 查询，不会触发数据从 SSD 回填到内存
- 如果要观测 SSD 读盘路径，应该重点关注 `get`

## 注意事项

- `spill_directory` 是本地冷层，不是持久化存储；Worker 重启时会清理 spill 目录下的 `datasystem_spill_data`
- 达到 `spill_size_limit` 后，当前这次 spill 不保证同步等待 SSD 淘汰完成后重试
- `access.log` 记录的是整次 `get` 的总耗时，不会单独拆出 SSD 读盘耗时
- 如果需要聚合统计形式的 spill 读盘时间，需要使用开启 `ENABLE_PERF` 编译的 Yuanrong 版本
