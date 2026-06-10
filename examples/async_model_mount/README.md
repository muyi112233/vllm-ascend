# Async Model Mount Mock

This example provides a local `model_manager` mock package for testing
`VLLM_ASCEND_ASYNC_MODEL_MOUNT` without a real model-manager service.

## Files

- `mock_model_manager/model_manager/apis.py`: implements
  `get_model_path_from_manager(with_weights: bool)`.
- `prepare_meta_only_model.py`: copies model metadata to a bootstrap model
  directory while excluding weight files.

## Prepare a Meta-Only Bootstrap Directory

```bash
python examples/async_model_mount/prepare_meta_only_model.py \
  --source /mnt/cephfs/models/GLM-5-w4a8 \
  --target /tmp/async_mount/bootstrap_model
```

The helper excludes common weight files such as `*.safetensors`, `*.bin`,
`*.pt`, `*.pth`, and `*.gguf`. It keeps config, tokenizer, index, quantization
metadata, templates, and small runtime metadata directories.

## Run with the Mock Model Manager

```bash
export PYTHONPATH="$(pwd)/examples/async_model_mount/mock_model_manager:${PYTHONPATH}"
export VLLM_ASCEND_ASYNC_MODEL_MOUNT=1

export MOCK_MODEL_META_PATH=/tmp/async_mount/bootstrap_model
export MOCK_MODEL_WEIGHT_PATH=/tmp/async_mount/mounted_model
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
When `with_weights=True`, it waits for `MOCK_MODEL_READY_FILE` if configured,
then returns `MOCK_MODEL_WEIGHT_PATH`.

## Environment Variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `MOCK_MODEL_META_PATH` | Yes | unset | Path returned for metadata-only access. |
| `MOCK_MODEL_WEIGHT_PATH` | Yes | unset | Path returned when weights are required. |
| `MOCK_MODEL_READY_FILE` | No | unset | If set, `with_weights=True` waits for this file. |
| `MOCK_MODEL_MOUNT_TIMEOUT_SEC` | No | `600` | Max wait time for the ready file. |
| `MOCK_MODEL_MOUNT_POLL_INTERVAL_SEC` | No | `1` | Ready-file polling interval. |
| `MOCK_MODEL_VALIDATE_PATH` | No | `1` | Validate returned paths exist when set to `1`. |
