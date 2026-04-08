import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MooncakeConnectorMetadata,
    _chunk_transfer_slices,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.yuanrong_connector import (
    YuanrongConnector,
    YuanrongConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.utils.yuanrong_transfer_engine import (
    GlobalYuanrongTE,
    YuanrongTransferEngineAdapter,
    global_yuanrong_te,
)


class FakeResult:
    def __init__(self, error: bool = False, msg: str = "ok"):
        self._error = error
        self._msg = msg

    def is_error(self) -> bool:
        return self._error

    def to_string(self) -> str:
        return self._msg


class TestYuanrongTransferEngineAdapter(unittest.TestCase):
    def test_adapter_maps_result_to_mooncake_style_status(self):
        raw_engine = MagicMock()
        raw_engine.get_rpc_port.return_value = 62000
        raw_engine.register_memory.return_value = FakeResult(error=False)
        raw_engine.batch_transfer_sync_read.return_value = FakeResult(error=False)

        adapter = YuanrongTransferEngineAdapter(raw_engine)

        self.assertEqual(adapter.get_rpc_port(), 62000)
        self.assertEqual(adapter.register_memory(0x1000, 128), 0)
        self.assertEqual(
            adapter.batch_transfer_sync_read("127.0.0.1:62000", [0x2000], [0x3000], [256]),
            0,
        )

        raw_engine.batch_transfer_sync_read.return_value = FakeResult(error=True, msg="transfer failed")
        self.assertEqual(
            adapter.batch_transfer_sync_read("127.0.0.1:62000", [0x2000], [0x3000], [256]),
            -1,
        )


class TestGlobalYuanrongTE(unittest.TestCase):
    def test_get_transfer_engine_initializes_with_current_device(self):
        raw_engine = MagicMock()
        raw_engine.initialize.return_value = FakeResult(error=False)
        raw_engine.get_rpc_port.return_value = 62000

        fake_yr = types.ModuleType("yr")
        fake_datasystem = types.ModuleType("yr.datasystem")
        fake_datasystem.TransferEngine = MagicMock(return_value=raw_engine)

        te = GlobalYuanrongTE()
        with patch.dict(sys.modules, {"yr": fake_yr, "yr.datasystem": fake_datasystem}):
            with patch(
                "vllm_ascend.distributed.kv_transfer.utils.yuanrong_transfer_engine._build_local_endpoint",
                return_value="127.0.0.1:62000",
            ), patch(
                "vllm_ascend.distributed.kv_transfer.utils.yuanrong_transfer_engine._get_default_device_name",
                return_value="npu:3",
            ):
                adapter = te.get_transfer_engine("127.0.0.1", device_name=None)

        raw_engine.initialize.assert_called_once_with("127.0.0.1:62000", "ascend", "npu:3")
        self.assertIsInstance(adapter, YuanrongTransferEngineAdapter)
        self.assertEqual(adapter.get_rpc_port(), 62000)

    def test_register_buffer_is_idempotent(self):
        te = GlobalYuanrongTE()
        te.transfer_engine = MagicMock()
        te.transfer_engine.register_memory.return_value = 0

        te.register_buffer([0x1000, 0x2000], [64, 128])
        te.register_buffer([0x1000, 0x2000], [64, 128])

        self.assertEqual(te.transfer_engine.register_memory.call_count, 2)


class TestYuanrongConnector(unittest.TestCase):
    def test_chunk_transfer_slices_splits_by_item_limit(self):
        chunks = list(
            _chunk_transfer_slices(
                [1, 2, 3, 4, 5],
                [11, 12, 13, 14, 15],
                [10, 10, 10, 10, 10],
                max_items_per_batch=2,
                max_bytes_per_batch=100,
            )
        )

        self.assertEqual(
            chunks,
            [
                ([1, 2], [11, 12], [10, 10]),
                ([3, 4], [13, 14], [10, 10]),
                ([5], [15], [10]),
            ],
        )

    def test_chunk_transfer_slices_splits_by_byte_limit(self):
        chunks = list(
            _chunk_transfer_slices(
                [1, 2, 3],
                [11, 12, 13],
                [40, 50, 30],
                max_items_per_batch=10,
                max_bytes_per_batch=80,
            )
        )

        self.assertEqual(
            chunks,
            [
                ([1], [11], [40]),
                ([2, 3], [12, 13], [50, 30]),
            ],
        )

    def test_worker_role_uses_yuanrong_transfer_engine_global(self):
        self.assertIs(YuanrongConnectorWorker.transfer_engine_global, global_yuanrong_te)

    def test_connector_instantiates_configured_worker(self):
        vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="engine-1"))
        mock_worker = MagicMock()
        worker_cls = MagicMock(return_value=mock_worker)

        with patch.object(YuanrongConnector, "worker_cls", worker_cls):
            connector = YuanrongConnector(vllm_config, KVConnectorRole.WORKER)

        worker_cls.assert_called_once_with(vllm_config, "engine-1")
        self.assertIs(connector.connector_worker, mock_worker)
        self.assertIsNone(connector.connector_scheduler)
        self.assertIsInstance(connector._connector_metadata, MooncakeConnectorMetadata)


if __name__ == "__main__":
    unittest.main()
