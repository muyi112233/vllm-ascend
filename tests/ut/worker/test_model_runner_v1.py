import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))

    def test_finalize_deferred_kv_connector_output_waits_for_save(self):
        runner = self._build_runner()
        scheduler_output = SimpleNamespace(finished_req_ids={"req-1"})
        kv_connector_output = SimpleNamespace(
            finished_sending=[],
            finished_recving=[],
            invalid_block_ids=None,
            kv_connector_stats=None,
            kv_cache_events=None,
        )
        kv_connector = MagicMock()
        kv_connector.get_finished.return_value = ({"req-1"}, set())
        kv_connector.get_block_ids_with_load_errors.return_value = {3}
        kv_connector.get_kv_connector_stats.return_value = {"save": 1}
        kv_connector.get_kv_connector_kv_cache_events.return_value = ["event"]

        with (
            patch("vllm_ascend.worker.model_runner_v1.has_kv_transfer_group", return_value=True),
            patch("vllm_ascend.worker.model_runner_v1.get_kv_transfer_group", return_value=kv_connector),
        ):
            result = runner._finalize_deferred_kv_connector_output(
                scheduler_output,
                kv_connector_output,
            )

        self.assertIs(result, kv_connector_output)
        kv_connector.wait_for_save.assert_called_once_with()
        kv_connector.get_finished.assert_called_once_with({"req-1"})
        kv_connector.clear_connector_metadata.assert_called_once_with()
        self.assertEqual(result.finished_sending, {"req-1"})
        self.assertEqual(result.finished_recving, set())
        self.assertEqual(result.invalid_block_ids, {3})
        self.assertEqual(result.kv_connector_stats, {"save": 1})
        self.assertEqual(result.kv_cache_events, ["event"])


if __name__ == "__main__":
    unittest.main()
