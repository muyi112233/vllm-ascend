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

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.model_loader.rfork.rfork_loader import RForkModelLoader


def test_ensure_rfork_worker_does_not_prefetch_fallback_path_for_existing_worker():
    rfork_worker = MagicMock()
    loader = RForkModelLoader.__new__(RForkModelLoader)
    loader.load_config = SimpleNamespace(rfork_worker=rfork_worker)

    assert loader._ensure_rfork_worker(SimpleNamespace()) is rfork_worker
    rfork_worker.prefetch_fallback_model_path.assert_not_called()
