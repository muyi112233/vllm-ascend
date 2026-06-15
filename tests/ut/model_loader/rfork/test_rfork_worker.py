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

import threading
from unittest.mock import MagicMock

import pytest

from vllm_ascend.model_loader.rfork.rfork_worker import RForkWorker


def _make_worker():
    worker = RForkWorker.__new__(RForkWorker)
    worker.fallback_model_path = None
    worker._fallback_path_thread = None
    worker._fallback_path_error = None
    worker._fallback_path_lock = threading.Lock()
    worker.seed_protocol = MagicMock()
    return worker


def test_get_fallback_model_path_resolves_without_prefetch(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ASYNC_MODEL_MOUNT", "1")
    monkeypatch.setattr(
        "vllm_ascend.model_loader.async_mount.get_model_path",
        lambda with_weights: "/mounted/model",
    )
    worker = _make_worker()

    assert worker.get_fallback_model_path() == "/mounted/model"


def test_prefetch_fallback_model_path_resolves_in_background(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ASYNC_MODEL_MOUNT", "1")
    monkeypatch.setattr(
        "vllm_ascend.model_loader.async_mount.get_model_path",
        lambda with_weights: "/mounted/model",
    )
    worker = _make_worker()

    worker.prefetch_fallback_model_path()

    assert worker.get_fallback_model_path() == "/mounted/model"


def test_get_fallback_model_path_propagates_prefetch_error(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ASYNC_MODEL_MOUNT", "1")

    def raise_error(with_weights):
        raise ImportError("missing model-manager")

    monkeypatch.setattr("vllm_ascend.model_loader.async_mount.get_model_path", raise_error)
    worker = _make_worker()

    worker.prefetch_fallback_model_path()

    with pytest.raises(RuntimeError, match="Failed to resolve RFork fallback model path"):
        worker.get_fallback_model_path()


def test_prefetch_fallback_model_path_skips_when_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ASYNC_MODEL_MOUNT", "0")
    worker = _make_worker()

    worker.prefetch_fallback_model_path()

    assert worker._fallback_path_thread is None
    assert worker.get_fallback_model_path() is None


def test_wait_for_seed_or_fallback_model_path_returns_fallback_when_async_mount_wins(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ASYNC_MODEL_MOUNT", "1")
    monkeypatch.setattr(
        "vllm_ascend.model_loader.async_mount.get_model_path",
        lambda with_weights: "/mounted/model",
    )
    worker = _make_worker()
    worker.seed_protocol.get_seed.return_value = None

    result = worker.wait_for_seed_or_fallback_model_path(retry_interval_sec=0.01)

    assert result.seed_available is False
    assert result.fallback_model_path == "/mounted/model"


def test_wait_for_seed_or_fallback_model_path_returns_seed_when_retry_wins(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ASYNC_MODEL_MOUNT", "1")
    async_mount_ready = threading.Event()

    def wait_for_model_path(with_weights):
        async_mount_ready.wait(timeout=1)
        return "/mounted/model"

    monkeypatch.setattr(
        "vllm_ascend.model_loader.async_mount.get_model_path",
        wait_for_model_path,
    )
    worker = _make_worker()
    seed = {
        "seed_ip": "127.0.0.1",
        "seed_port": "12345",
        "user_id": "user",
        "seed_rank": "0",
    }
    worker.seed_protocol.get_seed.return_value = seed

    result = worker.wait_for_seed_or_fallback_model_path(retry_interval_sec=0.01)
    async_mount_ready.set()
    worker._fallback_path_thread.join(timeout=1)

    assert result.seed_available is True
    assert result.fallback_model_path is None
    assert worker.rfork_seed == seed
