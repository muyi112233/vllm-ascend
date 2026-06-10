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

import pytest

from vllm_ascend.model_loader import async_mount


def test_get_model_path_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ASYNC_MODEL_MOUNT", "0")

    assert async_mount.get_model_path(with_weights=True) is None


def test_get_model_path_uses_model_manager_when_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ASYNC_MODEL_MOUNT", "1")
    monkeypatch.setattr(
        async_mount,
        "_get_model_manager_api",
        lambda: lambda with_weights: "/mounted/weights" if with_weights else "/mounted/config",
    )

    assert async_mount.get_model_path(with_weights=True) == "/mounted/weights"
    assert async_mount.get_model_path(with_weights=False) == "/mounted/config"


def test_link_model_at_refuses_existing_non_symlink(monkeypatch, tmp_path):
    target_path = tmp_path / "target"
    target_path.mkdir()
    link_path = tmp_path / "existing"
    link_path.mkdir()

    monkeypatch.setattr(async_mount, "get_model_path", lambda with_weights: str(target_path))

    with pytest.raises(FileExistsError):
        async_mount.link_model_at(str(link_path), with_weights=False)
