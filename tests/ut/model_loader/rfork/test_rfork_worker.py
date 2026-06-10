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

import pytest

from vllm_ascend.model_loader.rfork.rfork_worker import RForkWorker


def _make_worker():
    worker = RForkWorker.__new__(RForkWorker)
    worker.fallback_model_path = None
    worker._fallback_path_thread = None
    worker._fallback_path_error = None
    worker._fallback_path_lock = threading.Lock()
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
