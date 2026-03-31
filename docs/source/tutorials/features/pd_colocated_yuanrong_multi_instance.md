# PD-Colocated with Yuanrong Multi-Instance

## Getting Started

vLLM-Ascend now supports PD-colocated deployment with Yuanrong Datasystem features.
This guide provides step-by-step instructions to test these features with
constrained resources.

Using the Qwen2.5-72B-Instruct model as an example, this guide demonstrates
how to use vllm-ascend on two Atlas 800T A2 nodes to deploy two vLLM instances.
Each instance occupies 4 NPU cards and uses PD-colocated deployment.

## Verify Multi-Node Communication Environment

### Physical Layer Requirements

- The two Atlas 800T A2 nodes must be physically interconnected via a RoCE
  network. Without RoCE interconnection, cross-node KV Cache access
  performance will be significantly degraded.
- All NPU cards must communicate properly. Intra-node communication uses HCCS,
  while inter-node communication uses the RoCE network.

### Verification Process

The following process serves as a reference example. Please modify parameters
such as IP addresses according to your actual environment.

1. Single Node Verification:

   Execute the following commands sequentially. The results must all be
   `success` and the status must be `UP`:

   ```bash
   # Check the remote switch ports
   for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done
   # Get the link status of the Ethernet ports (UP or DOWN)
   for i in {0..7}; do hccn_tool -i $i -link -g ; done
   # Check the network health status
   for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
   # View the network detected IP configuration
   for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
   # View gateway configuration
   for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
   ```

2. Check NPU HCCN Configuration:

   Ensure that the hccn.conf file exists in the environment. If using Docker,
   mount it into the container.

   ```bash
   cat /etc/hccn.conf
   ```

3. Get NPU IP Addresses:

   ```bash
   for i in {0..7}; do hccn_tool -i $i -ip -g; done
   ```

4. Cross-Node PING Test:

   ```bash
   # Execute the following command on each node, replacing x.x.x.x
   # with the target node's NPU card address.
   for i in {0..7}; do hccn_tool -i $i -ping -g address x.x.x.x; done
   ```

5. Check NPU TLS Configuration

   ```bash
   # The tls settings should be consistent across all nodes.
   for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
   ```

## Run with Docker

Start a Docker container on each node.

```bash
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:latest
export NAME=vllm-ascend

# Run the container using the defined variables
# This test uses four NPU cards to create the container.
# Mount the hccn.conf file from the host node into the container.
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:\
/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

## Install Yuanrong Datasystem

Yuanrong Datasystem is a distributed data management system that provides
efficient KV Cache sharing across nodes.

### Install via pip

```bash
pip install openyuanrong-datasystem
```

Verify the installation:

```bash
python -c "import yr.datasystem; print('Yuanrong Datasystem installed successfully')"
```

For more information, refer to the official documentation:
<https://atomgit.com/openeuler/yuanrong-datasystem>

## Start etcd

Yuanrong Datasystem requires etcd for service discovery. Start a single-node
etcd cluster on one of the nodes (or a dedicated management node):

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

For production environments, refer to the official etcd clustering
documentation: <https://etcd.io/docs/current/op-guide/clustering/>

## Start Datasystem Worker

Start a Datasystem worker on each node using `dscli`. The worker provides
local storage and manages KV Cache data.

### On Node 1

```bash
export WORKER_IP="<node1_ip_address>"
export ETCD_IP="<etcd_server_ip>"

dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --etcd_address "${ETCD_IP}:2379" \
  --shared_memory_size_mb 102400
```

### On Node 2

```bash
export WORKER_IP="<node2_ip_address>"
export ETCD_IP="<etcd_server_ip>"

dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --etcd_address "${ETCD_IP}:2379" \
  --shared_memory_size_mb 102400
```

| Parameter              | Value                  | Explanation                           |
| ---------------------- | ---------------------- | ------------------------------------- |
| worker_address         | ${WORKER_IP}:31501     | Worker service address and port       |
| etcd_address           | ${ETCD_IP}:2379        | etcd service address for discovery    |
| shared_memory_size_mb  | 102400                 | Shared memory size (100 GB)           |

To stop a worker:

```bash
dscli stop --worker_address "${WORKER_IP}:31501"
```

## Environment Variable Configuration

Set the following environment variables on each node. The `DS_WORKER_ADDR`
must match the `dscli start --worker_address` address on the same host.

```bash
export PYTHONHASHSEED=0
# Required: must match local dscli --worker_address
export DS_WORKER_ADDR="${WORKER_IP}:31501"
# Optional (default: 0)
export DS_ENABLE_EXCLUSIVE_CONNECTION=0
export DS_ENABLE_REMOTE_H2D=0
```

| Parameter                    | Value              | Explanation                           |
| ---------------------------- | ------------------ | ------------------------------------- |
| PYTHONHASHSEED               | 0                  | Ensures consistent hash generation    |
| DS_WORKER_ADDR               | ${WORKER_IP}:31501 | Local worker address (required)       |
| DS_ENABLE_EXCLUSIVE_CONNECTION | 0                | Enable exclusive connection mode      |
| DS_ENABLE_REMOTE_H2D         | 0                  | Enable remote host-to-device transfer |

## vLLM Instance Deployment

Create containers on both Node 1 and Node 2, and launch the
Qwen2.5-72B-Instruct model service in each to test the reusability and
performance of cross-node, cross-instance KV Cache. Instance 1 utilizes NPU
cards [0-3] on the first Atlas 800T A2 server, while Instance 2 utilizes
cards [0-3] on the second server.

### Deploy Instance 1

Replace file paths, host, and port parameters based on your actual environment
configuration.

```bash
export PYTHONHASHSEED=0
export DS_WORKER_ADDR="<node1_ip_address>:31501"
export DS_ENABLE_EXCLUSIVE_CONNECTION=0
export DS_ENABLE_REMOTE_H2D=0

vllm serve <path_to_your_model>/Qwen2.5-72B-Instruct/ \
--served-model-name qwen \
--dtype bfloat16 \
--max-model-len 25600 \
--tensor-parallel-size 4 \
--host <node1_ip_address> \
--port 8002 \
--max-num-batched-tokens 4096 \
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

### Deploy Instance 2

The deployment method for Instance 2 is identical to Instance 1. Simply
modify the `--host`, `--port`, and `DS_WORKER_ADDR` parameters according to
your Instance 2 configuration.

```bash
export PYTHONHASHSEED=0
export DS_WORKER_ADDR="<node2_ip_address>:31501"
export DS_ENABLE_EXCLUSIVE_CONNECTION=0
export DS_ENABLE_REMOTE_H2D=0

vllm serve <path_to_your_model>/Qwen2.5-72B-Instruct/ \
--served-model-name qwen \
--dtype bfloat16 \
--max-model-len 25600 \
--tensor-parallel-size 4 \
--host <node2_ip_address> \
--port 8003 \
--max-num-batched-tokens 4096 \
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

### Configuration Parameters

| Parameter         | Value                 | Explanation                      |
| ----------------- | ----------------------| -------------------------------- |
| kv_connector      | AscendStoreConnector  | Use AscendStoreConnector         |
| kv_role           | kv_both               | Enable both produce and consume  |
| backend           | yuanrong              | Use Yuanrong backend             |
| lookup_rpc_port   | 0                     | Automatic port assignment        |
| load_async        | true                  | Enable asynchronous loading      |

## Benchmark

We recommend using the **AISBench** tool to assess performance. The test uses
**Dataset A**, consisting of fully random data, with the following
configuration:

- Input/output tokens: 1024/10
- Total requests: 100
- Concurrency: 25

The test procedure consists of three steps:

### Step 1: Baseline (No Cache)

Send Dataset A to Instance 1 on Node 1 and record the Time to First Token
(TTFT) as **TTFT1**.

### Preparation for Step 2

Before Step 2, send a fully random Dataset B to Instance 1. Due to the
unified HBM/DRAM KV Cache with LRU (Least Recently Used) eviction policy,
Dataset B's cache evicts Dataset A's cache from HBM, leaving Dataset A's
cache only in Node 1's DRAM.

### Step 2: Local DRAM Hit

Send Dataset A to Instance 1 again to measure the performance when hitting
the KV Cache in local DRAM. Record the TTFT as **TTFT2**.

### Step 3: Cross-Node DRAM Hit

Send Dataset A to Instance 2. With the Yuanrong KV Cache pool, this results
in a cross-node KV Cache hit from Node 1's DRAM. Record the TTFT as
**TTFT3**.

**Model Configuration**:

```python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="<path_to_your_model>/Qwen2.5-72B-Instruct",
        model="qwen",
        request_rate = 0,
        retry = 2,
        host_ip = "<your_server_ip>",
        host_port = 8002,
        max_out_len = 10,
        batch_size= 25,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            ignore_eos = True,
        ),
    )
]
```

**Performance Benchmarking Commands**:

```shell
ais_bench --models vllm_api_stream_chat \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --debug --summarizer default_perf --mode perf
```

### Expected Test Results

| Requests | Concur | TTFT1 (ms) | TTFT2 (ms) | TTFT3 (ms) |
| -------- | ------ | ---------- | ---------- | ---------- |
| 100      | 25     | ~2300      | ~700       | ~900       |

**Note**: Actual results may vary based on hardware configuration, network
bandwidth, and model size.

## Troubleshooting

### Common Issues

1. **Connection Failed to etcd**

   Ensure etcd is running and accessible from all nodes:
   ```bash
   etcdctl --endpoints "${ETCD_IP}:2379" endpoint health
   ```

2. **Worker Registration Failed**

   Check if the worker address is correctly configured and the port is not
   occupied:
   ```bash
   netstat -tlnp | grep 31501
   ```

3. **KV Cache Not Found**

   Verify that `PYTHONHASHSEED` is set to the same value on all nodes to
   ensure consistent key generation.

4. **Import Error for yr.datasystem**

   Ensure `openyuanrong-datasystem` is installed:
   ```bash
   pip install openyuanrong-datasystem
   python -c "from yr.datasystem.hetero_client import HeteroClient; print('OK')"
   ```

### Logs

Check vLLM logs for detailed error messages:
```bash
# Logs are typically output to stdout/stderr
# Redirect to file for analysis
vllm serve ... > vllm.log 2>&1
```

## References

- [Yuanrong Datasystem Documentation](https://atomgit.com/openeuler/yuanrong-datasystem)
- [etcd Documentation](https://etcd.io/docs/)
- [vLLM Ascend Documentation](https://docs.vllm.ai/projects/ascend/)
