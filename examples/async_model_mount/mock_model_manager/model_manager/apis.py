#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Mock model-manager API used by VLLM_ASCEND_ASYNC_MODEL_MOUNT examples."""

import os
import time
from pathlib import Path


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} must be set for the mock model-manager.")
    return value


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed_value = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float value, got {value!r}.") from exc
    if parsed_value <= 0:
        raise RuntimeError(f"{name} must be greater than 0, got {value!r}.")
    return parsed_value


def _get_non_negative_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed_value = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float value, got {value!r}.") from exc
    if parsed_value < 0:
        raise RuntimeError(f"{name} must be greater than or equal to 0, got {value!r}.")
    return parsed_value


def _validate_path(path: str, purpose: str) -> None:
    if os.getenv("MOCK_MODEL_VALIDATE_PATH", "1") != "1":
        return
    if not Path(path).exists():
        raise FileNotFoundError(f"Mock {purpose} path does not exist: {path}")


def _wait_until_ready() -> None:
    ready_file = os.getenv("MOCK_MODEL_READY_FILE")
    if not ready_file:
        return

    timeout_sec = _get_float_env("MOCK_MODEL_MOUNT_TIMEOUT_SEC", 600.0)
    poll_interval_sec = _get_float_env("MOCK_MODEL_MOUNT_POLL_INTERVAL_SEC", 1.0)
    deadline = time.monotonic() + timeout_sec

    while not Path(ready_file).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for mock model mount ready file: {ready_file}")
        time.sleep(poll_interval_sec)


def _sleep_until_mounted() -> None:
    delay_sec = _get_non_negative_float_env("MOCK_MODEL_MOUNT_DELAY_SEC", 0.0)
    if delay_sec > 0:
        time.sleep(delay_sec)


def get_model_path_from_manager(with_weights: bool) -> str:
    """Return the model path expected by vllm-ascend async mount code.

    Args:
        with_weights: If False, return the metadata/bootstrap model path. If
            True, wait for the configured mock mount delay and optional ready
            file, then return the full model path that contains weights.
    """
    if not with_weights:
        meta_path = _get_required_env("MOCK_MODEL_META_PATH")
        _validate_path(meta_path, "metadata")
        return meta_path

    _sleep_until_mounted()
    _wait_until_ready()
    weight_path = _get_required_env("MOCK_MODEL_WEIGHT_PATH")
    _validate_path(weight_path, "weight")
    return weight_path
