# Prefill-Decode Disaggregation with YuanrongConnector

## Getting Started

`YuanrongConnector` provides a direct KV transfer path for PD disaggregation by
using `yr.datasystem.TransferEngine` as the transport substrate.

This connector is intended for the same usage pattern as
`MooncakeConnectorV1`:

- the prefill instance acts as `kv_producer`
- the decode instance acts as `kv_consumer`
- KV cache blocks are transferred directly between the two instances

This is **not** the same feature as
`AscendStoreConnector + backend=yuanrong`, which is used for KV Pool and prefix
cache reuse.

This guide uses a single Atlas 800T A2 server with two NPUs as an example and
deploys a simple `1P1D` topology.

## Prerequisites

- vLLM Ascend contains `YuanrongConnector`
- `yr.datasystem` is importable in the runtime environment
- the prefill and decode processes can reach each other over TCP
- the selected NPU devices have already passed the usual HCCN and driver checks

## Prepare Yuanrong TransferEngine

`YuanrongConnector` depends only on `yr.datasystem.TransferEngine`.
It does **not** require `etcd`, `dscli`, or Datasystem worker processes.

### Option A: install from package

```bash
pip install openyuanrong-datasystem
python3 -c "from yr.datasystem import TransferEngine; print('Yuanrong TransferEngine is ready')"
```

### Option B: build from a local `transfer_engine` source tree

If you are developing from a local checkout such as
`C:\Code\yuanrong-datasystem\transfer_engine`, build the Python package first
and expose it through `PYTHONPATH`.

```bash
export TRANSFER_ENGINE_SRC=/path/to/yuanrong-datasystem/transfer_engine
export TE_BUILD_DIR=/tmp/transfer-engine-build
export TE_PYTHON_DIR=/tmp/transfer-engine-python

bash "${TRANSFER_ENGINE_SRC}/scripts/build_python_artifacts.sh" \
  "$(which python3)" \
  "${TRANSFER_ENGINE_SRC}" \
  "${TE_BUILD_DIR}" \
  "${TE_PYTHON_DIR}"

export PYTHONPATH="${TE_PYTHON_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${TE_PYTHON_DIR}/lib:${LD_LIBRARY_PATH}"

python3 -c "from yr.datasystem import TransferEngine; print('Yuanrong TransferEngine is ready')"
```

## Connector Scope and Constraints

- `YuanrongConnector` currently follows the same scheduler and worker flow as
  `MooncakeConnectorV1`
- it is a block-based KV transfer connector, not a layerwise push connector
- if you need reverse-triggered layerwise overlap, keep using
  `MooncakeLayerwiseConnector`
- `kv_port` is the fixed side-channel handshake port
- the Yuanrong transfer engine RPC port is allocated automatically at runtime
- on multi-node deployments, allow both the configured `kv_port` range and the
  dynamically allocated transfer-engine RPC ports through the network policy

## Start the Prefill and Decode Instances

The examples below use:

- model: `Qwen2.5-7B-Instruct`
- prefill HTTP port: `13700`
- decode HTTP port: `13701`
- prefill handshake base port: `30000`
- decode handshake base port: `30100`

### Common environment variables

Set the common network environment variables before starting each process.

```bash
export HCCL_IF_IP=192.0.0.1
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
```

### Prefill instance

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /model/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 13700 \
  --served-model-name qwen25 \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{
    "kv_connector": "YuanrongConnector",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
      "prefill": {
        "dp_size": 1,
        "tp_size": 1
      },
      "decode": {
        "dp_size": 1,
        "tp_size": 1
      }
    }
  }'
```

### Decode instance

```bash
export ASCEND_RT_VISIBLE_DEVICES=1

vllm serve /model/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 13701 \
  --served-model-name qwen25 \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{
    "kv_connector": "YuanrongConnector",
    "kv_role": "kv_consumer",
    "kv_port": "30100",
    "engine_id": "1",
    "kv_connector_extra_config": {
      "prefill": {
        "dp_size": 1,
        "tp_size": 1
      },
      "decode": {
        "dp_size": 1,
        "tp_size": 1
      }
    }
  }'
```

## Start the PD Proxy

Use the same proxy entry that is used by the existing Mooncake PD example:
`examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`

```bash
python3 /workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 192.0.0.1 \
  --port 8080 \
  --prefiller-hosts 192.0.0.1 \
  --prefiller-port 13700 \
  --decoder-hosts 192.0.0.1 \
  --decoder-ports 13701
```

## Verify the Deployment

Send a request to the proxy endpoint:

```bash
curl http://192.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen25",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a short summary of prefilling and decoding in vLLM."}
    ],
    "max_tokens": 128,
    "temperature": 0
  }'
```

If the deployment is working correctly:

- the request is accepted by the proxy
- the prefill process generates remote KV blocks
- the decode process loads the KV blocks through `YuanrongConnector`
- the final response is returned from the decode instance

## Troubleshooting

### `ImportError: Please install openyuanrong-datasystem`

`yr.datasystem` is not visible to the current Python environment.
Either install the package or export the locally built package directory
through `PYTHONPATH`.

### `torch.npu is unavailable` or wrong device id

`YuanrongConnector` initializes `TransferEngine` with the current NPU device.
Set `ASCEND_RT_VISIBLE_DEVICES` correctly before starting each vLLM process.

### Handshake succeeds but transfer fails

Check the following:

- prefill and decode nodes can reach each other over TCP
- `kv_port` is open on both sides
- the dynamically allocated Yuanrong transfer-engine RPC port is not blocked
- the two sides are using compatible `yr.datasystem` builds

### Confusion with Yuanrong KV Pool deployment

If your config uses:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_connector_extra_config": {
    "backend": "yuanrong"
  }
}
```

that is the KV Pool path, not `YuanrongConnector`.

## References

- [Yuanrong TransferEngine Python API](https://atomgit.com/openeuler/yuanrong-datasystem)
- [Mooncake single-node PD tutorial](./pd_disaggregation_mooncake_single_node.md)
- [EPD disaggregation feature guide](../../user_guide/feature_guide/epd_disaggregation.md)
