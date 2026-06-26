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

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_planner_module():
    planner_path = Path(__file__).parents[4] / "examples" / "rfork" / "rfork_planner.py"
    spec = importlib.util.spec_from_file_location("rfork_planner_for_test", planner_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


planner = _load_planner_module()


def _make_store():
    return planner.Store(
        heartbeat_ttl_sec=60,
        default_resource_points=1,
        scheduler=planner.Scheduler(),
        time_fn=lambda: 1.0,
    )


def test_parse_add_seed_headers_matches_private_protocol():
    parsed = planner.parse_add_seed_headers(
        {
            "SEED_KEY": "seed-key",
            "SEED_IP": "127.0.0.1",
            "SEED_PORT": "8080",
        }
    )

    assert parsed.seed_key == "seed-key"
    assert parsed.seed_ip == "127.0.0.1"
    assert parsed.seed_port == 8080
    assert not hasattr(parsed, "seed_refcnt")
    assert not hasattr(parsed, "seed_rank")


def test_parse_put_seed_headers_matches_private_protocol():
    parsed = planner.parse_put_seed_headers(
        {
            "USER_ID": "lease-id",
            "SEED_IP": "127.0.0.1",
            "SEED_PORT": "8080",
        }
    )

    assert parsed.user_id == "lease-id"
    assert parsed.seed_ip == "127.0.0.1"
    assert parsed.seed_port == 8080


@pytest.mark.parametrize(
    "headers",
    [
        {"USER_ID": "lease-id"},
        {"USER_ID": "lease-id", "SEED_IP": "127.0.0.1"},
        {"USER_ID": "lease-id", "SEED_PORT": "8080"},
    ],
)
def test_parse_put_seed_headers_requires_private_protocol_headers(headers):
    with pytest.raises(planner.HeaderError):
        planner.parse_put_seed_headers(headers)


def test_store_moves_same_seed_endpoint_between_seed_keys():
    store = _make_store()

    store.add_seed(seed_key="key-a", seed_ip="127.0.0.1", seed_port=8080)
    store.add_seed(seed_key="key-b", seed_ip="127.0.0.1", seed_port=8080)

    assert store.debug_snapshot()["seed_count"] == 1

    assert store.get_seed(seed_key="key-a") is None
    seed_b, _ = store.get_seed(seed_key="key-b")

    assert seed_b.seed_key == "key-b"
    assert seed_b.identity == "127.0.0.1:8080"


def test_store_put_seed_releases_by_user_id_without_seed_rank():
    store = _make_store()
    seed = store.add_seed(seed_key="key", seed_ip="127.0.0.1", seed_port=8080)
    _, lease = store.get_seed(seed_key="key")

    assert seed.resource_used == 1
    assert store.put_seed(user_id=lease.user_id, seed_ip="127.0.0.1", seed_port=8080)
    assert seed.resource_used == 0
    assert store.debug_snapshot()["lease_count"] == 0


def test_store_put_seed_rejects_endpoint_mismatch():
    store = _make_store()
    seed = store.add_seed(seed_key="key", seed_ip="127.0.0.1", seed_port=8080)
    _, lease = store.get_seed(seed_key="key")

    assert not store.put_seed(user_id=lease.user_id, seed_ip="127.0.0.2", seed_port=8080)
    assert seed.resource_used == 1
    assert store.debug_snapshot()["lease_count"] == 1

    assert store.put_seed(user_id=lease.user_id, seed_ip="127.0.0.1", seed_port=8080)
    assert seed.resource_used == 0


def test_planner_update_endpoints_match_private_get_protocol():
    app = planner.create_app(planner.Settings())
    route_methods = {route.path: route.methods for route in app.routes if hasattr(route, "methods")}

    assert route_methods["/add_seed"] == {"GET"}
    assert route_methods["/put_seed"] == {"GET"}
