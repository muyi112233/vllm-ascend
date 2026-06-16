# Async Model Mount Mock

This example provides a local `model_manager` mock package for testing
`VLLM_ASCEND_ASYNC_MODEL_MOUNT` without a real model-manager service.

## Files

- `mock_model_manager/model_manager/apis.py`: implements
  `get_model_path_from_manager(with_weights: bool)`.

## Run with the Mock Model Manager

```bash
export PYTHONPATH="$(pwd)/examples/async_model_mount/mock_model_manager:${PYTHONPATH}"
export VLLM_ASCEND_ASYNC_MODEL_MOUNT=1

export MOCK_MODEL_META_PATH=/tmp/async_mount/bootstrap_model
export MOCK_MODEL_WEIGHT_PATH=/tmp/async_mount/mounted_model
export MOCK_MODEL_MOUNT_DELAY_SEC=0
export MOCK_MODEL_READY_FILE=/tmp/async_mount/READY

vllm serve /tmp/async_mount/bootstrap_model
```

In another terminal, simulate the async mount becoming ready:

```bash
mkdir -p /tmp/async_mount/mounted_model
cp -a /mnt/cephfs/models/GLM-5-w4a8/. /tmp/async_mount/mounted_model/
touch /tmp/async_mount/READY
```

When `with_weights=False`, the mock returns `MOCK_MODEL_META_PATH`.
When `with_weights=True`, it waits for `MOCK_MODEL_MOUNT_DELAY_SEC`, then waits
for `MOCK_MODEL_READY_FILE` if configured, and returns `MOCK_MODEL_WEIGHT_PATH`.

To simulate RFork seed winning the race, set `MOCK_MODEL_MOUNT_DELAY_SEC` longer
than the seed ready time. To simulate async mount winning the race, set it
shorter than the seed ready time or leave it at `0`. If you want the race to be
controlled only by this delay, leave `MOCK_MODEL_READY_FILE` unset or create the
ready file before starting vLLM.

## Environment Variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `MOCK_MODEL_META_PATH` | Yes | unset | Path returned for metadata-only access. |
| `MOCK_MODEL_WEIGHT_PATH` | Yes | unset | Path returned when weights are required. |
| `MOCK_MODEL_MOUNT_DELAY_SEC` | No | `0` | Sleep duration before returning the weight path. |
| `MOCK_MODEL_READY_FILE` | No | unset | If set, `with_weights=True` waits for this file. |
| `MOCK_MODEL_MOUNT_TIMEOUT_SEC` | No | `600` | Max wait time for the ready file. |
| `MOCK_MODEL_MOUNT_POLL_INTERVAL_SEC` | No | `1` | Ready-file polling interval. |
| `MOCK_MODEL_VALIDATE_PATH` | No | `1` | Validate returned paths exist when set to `1`. |
