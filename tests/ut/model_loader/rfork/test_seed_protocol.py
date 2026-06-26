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

import pytest

import vllm_ascend.model_loader.rfork.seed_protocol as seed_protocol
from vllm_ascend.model_loader.rfork.seed_protocol import RForkSeedProtocol


class StopHeartbeat(Exception):
    pass


def _make_protocol() -> RForkSeedProtocol:
    return RForkSeedProtocol(
        disaggregation_mode="kv_consumer",
        node_rank=0,
        tp_rank=1,
        scheduler_url="http://planner",
        model_url="/models/dsv4",
        model_deploy_strategy_name="decode",
    )


def test_get_seed_does_not_require_seed_rank(monkeypatch):
    captured = {}

    def fake_get(url, headers, timeout):
        captured.update({"url": url, "headers": headers, "timeout": timeout})
        return SimpleNamespace(
            status_code=200,
            headers={
                "SEED_IP": "127.0.0.1",
                "SEED_PORT": "8080",
                "USER_ID": "lease-id",
            },
        )

    monkeypatch.setattr(seed_protocol.requests, "get", fake_get)

    seed = _make_protocol().get_seed()

    assert seed == {
        "seed_ip": "127.0.0.1",
        "seed_port": "8080",
        "user_id": "lease-id",
    }
    assert captured["url"] == "http://planner/get_seed"
    assert "SEED_KEY" in captured["headers"]


def test_release_seed_does_not_send_seed_rank(monkeypatch):
    captured = {}

    def fake_get(url, headers, timeout):
        captured.update({"url": url, "headers": headers, "timeout": timeout})
        return SimpleNamespace(status_code=200, text="")

    monkeypatch.setattr(seed_protocol.requests, "get", fake_get)

    assert _make_protocol().release_seed(
        {
            "seed_ip": "127.0.0.1",
            "seed_port": "8080",
            "user_id": "lease-id",
        }
    )
    assert captured["url"] == "http://planner/put_seed"
    assert captured["headers"] == {
        "SEED_IP": "127.0.0.1",
        "SEED_PORT": "8080",
        "USER_ID": "lease-id",
    }


def test_report_seed_does_not_send_seed_rank(monkeypatch):
    captured = {}

    def fake_get(url, headers, timeout):
        captured.update({"url": url, "headers": headers, "timeout": timeout})
        return SimpleNamespace(status_code=200, text="")

    def stop_sleep(_):
        raise StopHeartbeat()

    monkeypatch.setattr(seed_protocol.requests, "get", fake_get)
    monkeypatch.setattr(seed_protocol, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(seed_protocol.time, "sleep", stop_sleep)

    with pytest.raises(StopHeartbeat):
        _make_protocol().report_seed(8080, sleep_interval=0)

    assert captured["url"] == "http://planner/add_seed"
    assert captured["headers"]["SEED_IP"] == "127.0.0.1"
    assert captured["headers"]["SEED_PORT"] == "8080"
    assert "SEED_REFCNT" not in captured["headers"]
    assert "SEED_RANK" not in captured["headers"]
