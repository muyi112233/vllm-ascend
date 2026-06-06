import torch
import pytest
from torch import nn

from vllm_ascend.model_loader.rfork import transfer_backend as transfer_backend_module
from vllm_ascend.model_loader.rfork.transfer_backend import (
    MAX_TRANSFER_CHUNK_BYTES,
    RForkTransferBackend,
    _block_contains_weight_ptr,
    _iter_transfer_chunks,
    _iter_transferable_tensors,
)


@pytest.fixture(autouse=True)
def allow_cpu_tensors_for_manifest_tests(monkeypatch):
    monkeypatch.setattr(
        transfer_backend_module,
        "_is_tensor_on_transfer_device",
        lambda tensor: True,
    )


class _RuntimeImpl:
    def __init__(self):
        self.W_UV = torch.ones(8)
        self.W_UK_T = torch.ones(9)


class _ModelWithRuntimeTensors(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 3))
        self.zero_weight = nn.Parameter(torch.empty(0))
        self.register_buffer("buffer_weight", torch.ones(4))
        self.runtime_weight = torch.ones(5)
        self.runtime_weight_list = [torch.ones(6), torch.empty(0)]
        self.runtime_weight_dict = {"expert": torch.ones(7)}
        self.runtime_alias = self.weight.data
        self.impl = _RuntimeImpl()
        self.quant_method = _RuntimeImpl()


class _FakeResult:
    def is_error(self):
        return False


class _FakeTransferEngine:
    def __init__(self):
        self.transfer_args = None

    def batch_transfer_sync_read(self, seed_session_id, client_ptr_list, seed_ptr_list, client_len_list):
        self.transfer_args = (
            seed_session_id,
            client_ptr_list,
            seed_ptr_list,
            client_len_list,
        )
        return _FakeResult()


def test_iter_transferable_tensors_includes_runtime_tensor_attrs():
    names = {name for name, _ in _iter_transferable_tensors(_ModelWithRuntimeTensors())}

    assert "weight" in names
    assert "buffer_weight" in names
    assert "runtime_weight" in names
    assert "runtime_weight_list.0" in names
    assert "runtime_weight_dict.expert" in names
    assert "impl.W_UV" in names
    assert "impl.W_UK_T" in names
    assert "quant_method.W_UV" not in names
    assert "quant_method.W_UK_T" not in names
    assert "runtime_alias" not in names
    assert "zero_weight" not in names
    assert "runtime_weight_list.1" not in names


def test_block_contains_weight_ptr_matches_ptr_inside_block():
    sorted_weight_ptrs = [100, 250, 400]

    assert _block_contains_weight_ptr(80, 30, sorted_weight_ptrs)
    assert _block_contains_weight_ptr(200, 100, sorted_weight_ptrs)
    assert not _block_contains_weight_ptr(101, 149, sorted_weight_ptrs)
    assert not _block_contains_weight_ptr(450, 50, sorted_weight_ptrs)


def test_iter_transfer_chunks_splits_by_size_and_weight_count():
    names = ["a", "b", "c"]
    seed_ptrs = [1, 2, 3]
    client_ptrs = [4, 5, 6]
    lengths = [MAX_TRANSFER_CHUNK_BYTES, 1, 1]

    chunks = list(_iter_transfer_chunks(names, seed_ptrs, client_ptrs, lengths))

    assert [chunk[0] for chunk in chunks] == [["a"], ["b", "c"]]


def test_recv_from_source_uses_transferable_tensor_manifest(monkeypatch):
    model = _ModelWithRuntimeTensors()
    tensors = dict(_iter_transferable_tensors(model))
    seed_weight_info = {
        name: (index + 1, tensor.numel(), tensor.element_size())
        for index, (name, tensor) in enumerate(tensors.items())
    }

    monkeypatch.setattr(
        transfer_backend_module,
        "get_remote_instance_transfer_engine_info",
        lambda seed_url, local_seed_key: ("seed-session", seed_weight_info),
    )

    transfer_engine = _FakeTransferEngine()
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = transfer_engine

    assert backend.recv_from_source(
        model=model,
        seed_instance_ip="127.0.0.1",
        seed_instance_service_port=1234,
        local_seed_key="seed-key",
    )

    assert transfer_engine.transfer_args is not None
    seed_session_id, client_ptr_list, seed_ptr_list, client_len_list = transfer_engine.transfer_args
    assert seed_session_id == "seed-session"
    assert len(client_ptr_list) == len(seed_weight_info)
    assert seed_ptr_list == [info[0] for info in seed_weight_info.values()]
    assert client_len_list == [
        tensor.numel() * tensor.element_size()
        for tensor in tensors.values()
    ]
