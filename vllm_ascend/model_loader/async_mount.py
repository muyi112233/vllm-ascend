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

"""Utilities for resolving model paths from an external model manager."""

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from vllm.logger import logger

import vllm_ascend.envs as envs_ascend


@lru_cache(maxsize=1)
def _get_model_manager_api() -> Callable[[bool], str]:
    try:
        from model_manager.apis import get_model_path_from_manager
    except ImportError as exc:
        raise ImportError(
            "model-manager package is required when "
            "VLLM_ASCEND_ASYNC_MODEL_MOUNT is enabled. "
            "Install it before enabling async model mount."
        ) from exc

    return get_model_path_from_manager


def get_model_path(with_weights: bool) -> str | None:
    """Get a model path from model-manager when async mount is enabled."""
    if not envs_ascend.VLLM_ASCEND_ASYNC_MODEL_MOUNT:
        return None

    model_path = _get_model_manager_api()(with_weights)
    if not model_path:
        logger.warning(
            "model-manager returned an empty model path, with_weights=%s",
            with_weights,
        )
        return None

    return model_path


def link_model_at(link_path: str, with_weights: bool) -> str | None:
    """Create or update a symlink to the resolved model path.

    Existing non-symlink paths are never removed. This keeps async mount from
    deleting a user-provided model directory or file when the launch path is
    already populated.
    """
    model_path = get_model_path(with_weights)
    if not model_path:
        return None

    target = Path(model_path).resolve()
    link_name = Path(link_path)

    if link_name.is_symlink():
        if link_name.resolve() == target:
            return str(link_name)
        link_name.unlink()
    elif link_name.exists():
        if link_name.resolve() == target:
            return str(link_name)
        raise FileExistsError(
            f"Cannot create async model mount symlink at {link_name}: "
            "path already exists and is not a symlink."
        )

    link_name.parent.mkdir(parents=True, exist_ok=True)
    link_name.symlink_to(target, target_is_directory=target.is_dir())
    logger.info(
        "Created async model mount symlink: %s -> %s, with_weights=%s",
        link_name,
        target,
        with_weights,
    )
    return str(link_name)
