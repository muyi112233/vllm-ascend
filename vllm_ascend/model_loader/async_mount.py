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

    return str(model_path)
