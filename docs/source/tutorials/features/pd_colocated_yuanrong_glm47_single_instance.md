# GLM-4.7 Single Instance with Yuanrong KV Pool

## Getting Started

This guide provides step-by-step instructions to deploy GLM-4.7 model with
Yuanrong Datasystem as the KV Pool backend on a single Atlas 800T A2 node
with 4 NPU cards.

GLM-4.7 is a Mixture-of-Experts (MoE) architecture model specifically designed
for agent applications. Using Yuanrong as the KV Pool backend enables efficient
KV Cache storage and reuse across requests.

## Prerequisites

### Hardware Requirements

- 1 × Atlas 800T A2 node with 4 NPU cards (64G each)
- RoCE network configured for optimal performance

### Software Requirements

- Python >= 3.10, < 3.12
- CANN == 8.3.rc2 or later
- PyTorch == 2.8.0, torch-npu == 2.8.0
- vLLM main branch
- vLLM-Ascend main branch

### Model Weight

Download the GLM-4.7 model weight:

- `GLM-4.7` (BF16 version): [Download](https://www.modelscope.cn/models/ZhipuAI/GLM-4.7)
- `GLM-4.7-w8a8-with-float-mtp` (Quantized version): [Download](https://modelscope.cn/models/Eco-Tech/GLM-4.7-W8A8-floatmtp)

It is recommended to download the model weight to a shared directory, such as
`/root/.cache/`.

## Run with Docker

Start a Docker container with 4 NPU cards:

```bash
export IMAGE=quay.io/ascend/vllm-ascend:latest
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

## Install Yuanrong Datasystem

### Install via pip

```bash
pip install openyuanrong-datasystem
```

Verify the installation:

```bash
python -c "import yr.datasystem; print('Yuanrong Datasystem installed successfully')"
```

## Start etcd

Yuanrong Datasystem requires etcd for service discovery. Start etcd on the
same node or a dedicated management node:

```bash
ETCD_VERSION="v3.5.12"
ETCD_IP="<your_etcd_server_ip>"
if [ "$(uname -m)" = "aarch64" ]; then
  ETCD_ARCH="linux-arm64"
else
  ETCD_ARCH="linux-amd64"
fi
wget https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
tar -xvf etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
cd etcd-${ETCD_VERSION}-${ETCD_ARCH}
sudo cp etcd etcdctl /usr/local/bin/

etcd \
  --name etcd-single \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://${ETCD_IP}:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://${ETCD_IP}:2380 \
  --initial-cluster etcd-single=http://${ETCD_IP}:2380 &

# Verify etcd is running
etcdctl --endpoints "${ETCD_IP}:2379" put key "value"
etcdctl --endpoints "${ETCD_IP}:2379" get key
```

## Start Datasystem Worker

Start a Datasystem worker on the node:

```bash
export WORKER_IP="<your_node_ip_address>"
export ETCD_IP="<etcd_server_ip>"

dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --etcd_address "${ETCD_IP}:2379" \
  --shared_memory_size_mb 51200
```

| Parameter              | Value                  | Explanation                           |
| ---------------------- | ---------------------- | ------------------------------------- |
| worker_address         | ${WORKER_IP}:31501     | Worker service address and port       |
| etcd_address           | ${ETCD_IP}:2379        | etcd service address for discovery    |
| shared_memory_size_mb  | 51200                  | Shared memory size (50 GB)            |

To stop the worker:

```bash
dscli stop --worker_address "${WORKER_IP}:31501"
```

## Environment Variable Configuration

Set the following environment variables:

```bash
export PYTHONHASHSEED=0
# Required: must match local dscli --worker_address
export DS_WORKER_ADDR="${WORKER_IP}:31501"
# Optional (default: 0)
export DS_ENABLE_EXCLUSIVE_CONNECTION=0
export DS_ENABLE_REMOTE_H2D=0
```

## Deploy GLM-4.7 with Yuanrong KV Pool

### Option 1: GLM-4.7 BF16 Version

```bash
#!/bin/sh

# Network configuration
nic_name="xxxx"  # Change to your network interface name
local_ip="xxxx"  # Change to your node IP

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

# Yuanrong configuration
export PYTHONHASHSEED=0
export DS_WORKER_ADDR="${local_ip}:31501"
export DS_ENABLE_EXCLUSIVE_CONNECTION=0
export DS_ENABLE_REMOTE_H2D=0

vllm serve <path_to_your_model>/GLM-4.7 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name glm \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config '{
      "kv_connector": "AscendStoreConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
          "lookup_rpc_port": "0",
          "backend": "yuanrong",
          "load_async": true
      }
  }'
```

### Option 2: GLM-4.7 W8A8 Quantized Version (Recommended)

The quantized version provides better memory efficiency and allows longer
context lengths.

```bash
#!/bin/sh

# Network configuration
nic_name="xxxx"  # Change to your network interface name
local_ip="xxxx"  # Change to your node IP

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

# Yuanrong configuration
export PYTHONHASHSEED=0
export DS_WORKER_ADDR="${local_ip}:31501"
export DS_ENABLE_EXCLUSIVE_CONNECTION=0
export DS_ENABLE_REMOTE_H2D=0

vllm serve Eco-Tech/GLM-4.7-W8A8-floatmtp \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name glm \
  --max-model-len 65536 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --quantization ascend \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --speculative-config '{"num_speculative_tokens": 3, "model":"Eco-Tech/GLM-4.7-W8A8-floatmtp", "method":"mtp"}' \
  --compilation-config '{"cudagraph_capture_sizes": [1,2,4,8,16,32,64,128,256], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_shared_expert_dp": true, "ascend_fusion_config": {"fusion_ops_gmmswigluquant": false}}' \
  --kv-transfer-config '{
      "kv_connector": "AscendStoreConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
          "lookup_rpc_port": "0",
          "backend": "yuanrong",
          "load_async": true
      }
  }'
```

### Configuration Parameters

| Parameter                | Value                   | Explanation                           |
| ------------------------ | ----------------------- | ------------------------------------- |
| tensor-parallel-size     | 4                       | Use 4 NPU cards                       |
| enable-expert-parallel   | (flag)                  | Enable expert parallelism for MoE     |
| max-model-len            | 32768/65536             | Maximum context length                |
| kv_connector             | AscendStoreConnector    | Use AscendStoreConnector              |
| kv_role                  | kv_both                 | Enable both produce and consume       |
| backend                  | yuanrong                | Use Yuanrong backend                  |
| load_async               | true                    | Enable asynchronous loading           |

## Functional Verification

Once your server is started, verify the deployment:

### Check Server Status

```bash
curl http://localhost:8000/health
```

### Test Inference

```bash
curl -H "Accept: application/json" \
    -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "glm", 
        "messages": [{ 
            "role": "user", 
            "content": "Hello, what is the future of AI?" 
        }], 
        "stream": false, 
        "ignore_eos": false, 
        "temperature": 0, 
        "max_tokens": 200 
    }' http://localhost:8000/v1/chat/completions
```

### Test KV Cache Reuse

Send the same prompt twice to verify KV Cache is being reused:

```bash
# First request - should populate KV Cache
curl -H "Accept: application/json" \
    -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "glm", 
        "messages": [{ 
            "role": "user", 
            "content": "Please explain the concept of machine learning in detail." 
        }], 
        "stream": false, 
        "temperature": 0, 
        "max_tokens": 100 
    }' http://localhost:8000/v1/chat/completions

# Second request - should hit KV Cache (faster TTFT)
curl -H "Accept: application/json" \
    -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "glm", 
        "messages": [{ 
            "role": "user", 
            "content": "Please explain the concept of machine learning in detail." 
        }], 
        "stream": false, 
        "temperature": 0, 
        "max_tokens": 100 
    }' http://localhost:8000/v1/chat/completions
```

## Performance Benchmark

### Using AISBench

```shell
ais_bench --models vllm_api_stream_chat \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --debug --summarizer default_perf --mode perf
```

### Using vLLM Benchmark

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
  --model glm \
  --tokenizer Eco-Tech/GLM-4.7-W8A8-floatmtp \
  --seed 1000 \
  --host 0.0.0.0 \
  --port 8000 \
  --endpoint /v1/completions \
  --max-concurrency 4 \
  --request-rate 1
```

## Troubleshooting

### Common Issues

1. **Connection Failed to etcd**

   Ensure etcd is running and accessible:
   ```bash
   etcdctl --endpoints "${ETCD_IP}:2379" endpoint health
   ```

2. **Worker Registration Failed**

   Check if the worker address is correctly configured:
   ```bash
   netstat -tlnp | grep 31501
   ```

3. **KV Cache Not Found**

   Verify that `PYTHONHASHSEED` is set consistently:
   ```bash
   echo $PYTHONHASHSEED
   ```

4. **Import Error for yr.datasystem**

   Ensure `openyuanrong-datasystem` is installed:
   ```bash
   pip install openyuanrong-datasystem
   python -c "from yr.datasystem.hetero_client import HeteroClient; print('OK')"
   ```

5. **OOM (Out of Memory)**

   Reduce `--max-num-seqs` and `--max-model-len`:
   ```bash
   --max-model-len 16384 \
   --max-num-seqs 4
   ```

6. **FIA Operator Performance Issue**

   Execute the FIA operator replacement script:
   ```bash
   # For A2 series
   bash tools/install_flash_infer_attention_score_ops_a2.sh
   ```

### Logs

Check vLLM logs for detailed error messages:
```bash
vllm serve ... > vllm.log 2>&1
tail -f vllm.log
```

## References

- [Yuanrong Datasystem Documentation](https://atomgit.com/openeuler/yuanrong-datasystem)
- [etcd Documentation](https://etcd.io/docs/)
- [vLLM Ascend Documentation](https://docs.vllm.ai/projects/ascend/)
- [GLM-4.x Tutorial](../../tutorials/models/GLM4.x.md)
