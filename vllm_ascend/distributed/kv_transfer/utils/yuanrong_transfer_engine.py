import socket
import threading

import torch
from vllm.logger import logger
from vllm.utils.network_utils import split_host_port


def _allocate_local_port(hostname: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((hostname, 0))
        return int(sock.getsockname()[1])


def _build_local_endpoint(hostname: str) -> str:
    try:
        split_host_port(hostname)
        return hostname
    except Exception:
        return f"{hostname}:{_allocate_local_port(hostname)}"


def _get_default_device_name() -> str:
    if not hasattr(torch, "npu"):
        raise RuntimeError("torch.npu is unavailable, cannot initialize yuanrong transfer engine.")
    return f"npu:{int(torch.npu.current_device())}"


def _bind_npu_device(device_name: str) -> None:
    if not device_name.startswith("npu:"):
        return
    torch.npu.set_device(torch.device(device_name))


class YuanrongTransferEngineAdapter:
    def __init__(self, transfer_engine):
        self._transfer_engine = transfer_engine

    def get_rpc_port(self) -> int:
        return int(self._transfer_engine.get_rpc_port())

    def register_memory(self, ptr: int, size: int) -> int:
        result = self._transfer_engine.register_memory(ptr, size)
        if result.is_error():
            logger.error("Yuanrong register_memory failed: %s", result.to_string())
            return -1
        return 0

    def batch_transfer_sync_read(
        self,
        target_hostname: str,
        buffers: list[int],
        peer_buffer_addresses: list[int],
        lengths: list[int],
    ) -> int:
        result = self._transfer_engine.batch_transfer_sync_read(
            target_hostname,
            buffers,
            peer_buffer_addresses,
            lengths,
        )
        if result.is_error():
            logger.error("Yuanrong batch_transfer_sync_read failed: %s", result.to_string())
            return -1
        return 0


class GlobalYuanrongTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                if self.transfer_engine is None:
                    try:
                        from yr.datasystem import TransferEngine  # type: ignore[import-not-found]
                    except ImportError as e:
                        raise ImportError(
                            "Please install openyuanrong-datasystem to run vLLM with YuanrongConnector."
                        ) from e
                    engine = TransferEngine()
                    local_endpoint = _build_local_endpoint(hostname)
                    normalized_device_name = device_name or _get_default_device_name()
                    _bind_npu_device(normalized_device_name)
                    result = engine.initialize(local_endpoint, "ascend", normalized_device_name)
                    if result.is_error():
                        raise RuntimeError(
                            "Yuanrong TransferEngine initialization failed: "
                            f"{result.to_string()}"
                        )
                    self.transfer_engine = YuanrongTransferEngineAdapter(engine)
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Yuanrong memory registration failed.")
            self.is_register_buffer = True


global_yuanrong_te = GlobalYuanrongTE()
