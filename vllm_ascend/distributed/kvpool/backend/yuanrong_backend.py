import hashlib
import os
import re

import torch
from vllm.config import ParallelConfig
from vllm.logger import logger
from vllm.utils.network_utils import split_host_port

from vllm_ascend.distributed.kvpool.backend.backend import Backend


class YuanrongHelper:

    _DS_KEY_MAX_LEN = 255
    _DS_KEY_ALLOWED_PATTERN = re.compile(
        r"^[a-zA-Z0-9\-_!@#%\^\*\(\)\+\=\:;]+$")
    _DS_KEY_INVALID_CHAR_PATTERN = re.compile(
        r"[^a-zA-Z0-9\-_!@#%\^\*\(\)\+\=\:;]")
    _DS_KEY_HASH_SUFFIX_LEN = 16

    def __init__(self, blob_cls, blob_list_cls):
        self._blob_cls = blob_cls
        self._blob_list_cls = blob_list_cls
        self._device_id: int | None = None

    def normalize_keys(self, keys: list[str]) -> list[str]:
        normalized: list[str] = []
        for key in keys:
            if (len(key) <= self._DS_KEY_MAX_LEN
                    and self._DS_KEY_ALLOWED_PATTERN.match(key)):
                normalized.append(key)
                continue

            sanitized = self._DS_KEY_INVALID_CHAR_PATTERN.sub("_", key)
            hash_digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            suffix = f"__{hash_digest[:self._DS_KEY_HASH_SUFFIX_LEN]}"
            max_prefix_len = self._DS_KEY_MAX_LEN - len(suffix)
            normalized.append(sanitized[:max_prefix_len] + suffix)
        return normalized

    def make_blob_lists(
            self, addrs_list: list[list[int]],
            sizes_list: list[list[int]]) -> list["DeviceBlobList"]:
        total = len(addrs_list)
        if total != len(sizes_list):
            raise ValueError("Address list and size list length mismatch.")

        device_id = self._device_id
        if device_id is None:
            logger.error(
                "Device id is not set. Call set_device() before using the "
                "yuanrong backend.")
            raise RuntimeError("Yuanrong backend device id is not initialized.")

        blob_lists: list["DeviceBlobList"] = []
        for addrs, sizes in zip(addrs_list, sizes_list):
            if len(addrs) != len(sizes):
                raise ValueError(
                    "Address list and size list length mismatch.")
            blobs = [
                self._blob_cls(addr, size)  # type: ignore[misc]
                for addr, size in zip(addrs, sizes)
            ]
            blob_lists.append(
                self._blob_list_cls(device_id, blobs)  # type: ignore[misc]
            )
        return blob_lists


class YuanrongBackend(Backend):

    def __init__(self, parallel_config: ParallelConfig):
        try:
            from yr.datasystem.hetero_client import (  # type: ignore
                HeteroClient, Blob, DeviceBlobList)
            from yr.datasystem.kv_client import SetParam  # type: ignore
            from yr.datasystem.object_client import WriteMode  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Please install openyuanrong-datasystem to use the "
                "yuanrong datasystem backend.") from exc

        self.rank = parallel_config.rank
        self._helper = YuanrongHelper(Blob, DeviceBlobList)
        self._ds_set_param = SetParam()
        self._ds_set_param.write_mode = WriteMode.NONE_L2_CACHE_EVICT

        worker_addr = os.getenv("DS_WORKER_ADDR", "")
        host, port = split_host_port(worker_addr)
        enable_exclusive_connection = bool(
            int(os.getenv("DS_ENABLE_EXCLUSIVE_CONNECTION", "0")))
        enable_remote_h2d = bool(int(os.getenv("DS_ENABLE_REMOTE_H2D", "0")))
        self._hetero_client = HeteroClient(
            host,
            int(port),
            enable_exclusive_connection=enable_exclusive_connection,
            enable_remote_h2d=enable_remote_h2d,
        )
        self._hetero_client.init()

    def set_device(self):
        device = torch.device(f"npu:{self.rank}")
        torch.npu.set_device(device)
        self._helper._device_id = int(torch.npu.current_device())

    def exists(self, keys: list[str]) -> list[int]:
        if len(keys) == 0:
            return []
        try:
            keys = self._helper.normalize_keys(keys)
            exists = self._hetero_client.exist(keys)  # type: ignore[union-attr]
            return [1 if value else 0 for value in exists]
        except Exception as exc:
            logger.error("Failed to check keys %s: %s", keys, exc)
            return [0] * len(keys)

    def get(self, keys: list[str], addrs: list[list[int]],
            sizes: list[list[int]]):
        if len(keys) == 0:
            return
        try:
            keys = self._helper.normalize_keys(keys)
            blob_lists = self._helper.make_blob_lists(addrs, sizes)
            failed_keys = self._hetero_client.mget_h2d(  # type: ignore[union-attr]
                keys, blob_lists, 0)
            for key in failed_keys:
                logger.error("Failed to get key %s", key)
        except Exception as exc:
            logger.error("Failed to get keys %s: %s", keys, exc)

    def put(self, keys: list[str], addrs: list[list[int]],
            sizes: list[list[int]]):
        if len(keys) == 0:
            return
        try:
            keys = self._helper.normalize_keys(keys)
            blob_lists = self._helper.make_blob_lists(addrs, sizes)
            self._hetero_client.mset_d2h(  # type: ignore[union-attr]
                keys, blob_lists, self._ds_set_param)
        except Exception as exc:
            logger.error("Failed to put keys %s: %s", keys, exc)
