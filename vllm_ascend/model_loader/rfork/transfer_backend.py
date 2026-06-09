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

import hashlib
import time
from bisect import bisect_left
from typing import Any

import requests
import torch
from torch import nn
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, get_open_port, join_host_port

MAX_TRANSFER_CHUNK_BYTES = 1024**3
MAX_TRANSFER_CHUNK_WEIGHTS = 512
MAX_DEBUG_TENSOR_SAMPLE_ELEMENTS = 4096
MAX_DEBUG_TENSOR_VERIFY_COUNT = 32


def _is_transferable_tensor(tensor: torch.Tensor) -> bool:
    return not tensor.is_meta and tensor.numel() > 0 and _is_tensor_on_transfer_device(tensor)


def _is_tensor_on_transfer_device(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "npu"


def _get_tensor_transfer_region(tensor: torch.Tensor) -> tuple[int, int]:
    data_ptr = tensor.data_ptr()
    logical_nbytes = tensor.numel() * tensor.element_size()
    try:
        storage = tensor.untyped_storage()
        storage_ptr = storage.data_ptr()
        storage_nbytes = storage.nbytes()
        storage_offset = tensor.storage_offset()
    except Exception:
        return data_ptr, logical_nbytes

    if storage_ptr > 0 and storage_offset == 0:
        return storage_ptr, max(logical_nbytes, storage_nbytes)
    return data_ptr, logical_nbytes


def _iter_tensors_in_value(prefix: str, value: Any, visited_object_ids: set[int], scan_objects: bool = False):
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return

    if isinstance(value, (nn.Module, str, bytes)) or callable(value):
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensors_in_value(f"{prefix}.{index}", item, visited_object_ids, scan_objects)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors_in_value(f"{prefix}.{key}", item, visited_object_ids, scan_objects)
        return

    if not scan_objects or not hasattr(value, "__dict__"):
        return

    value_id = id(value)
    if value_id in visited_object_ids:
        return
    visited_object_ids.add(value_id)
    for attr_name, attr_value in vars(value).items():
        if attr_name.startswith("_"):
            continue
        yield from _iter_tensors_in_value(f"{prefix}.{attr_name}", attr_value, visited_object_ids, scan_objects)


def _iter_transferable_tensors(model: nn.Module):
    seen_regions: set[tuple[int, int]] = set()

    def should_yield(tensor: torch.Tensor) -> bool:
        if not _is_transferable_tensor(tensor):
            return False
        transfer_region = _get_tensor_transfer_region(tensor)
        if transfer_region in seen_regions:
            return False
        seen_regions.add(transfer_region)
        return True

    for name, tensor in model.named_parameters():
        if should_yield(tensor):
            yield name, tensor

    for name, tensor in model.named_buffers():
        if should_yield(tensor):
            yield name, tensor

    # Some Ascend post-load paths replace checkpoint parameters with runtime
    # tensors stored as plain module attributes, e.g. MLA/SFA W_UV and W_UK_T.
    for module_prefix, module in model.named_modules():
        for attr_name, attr_value in vars(module).items():
            if attr_name.startswith("_") or isinstance(attr_value, nn.Module):
                continue

            scan_objects = attr_name == "impl"
            for tensor_name, tensor in _iter_tensors_in_value(attr_name, attr_value, set(), scan_objects):
                if not should_yield(tensor):
                    continue

                full_name = f"{module_prefix}.{tensor_name}" if module_prefix else tensor_name
                yield full_name, tensor


def _normalize_debug_patterns(patterns: list[str] | str | None) -> list[str]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        return [pattern.strip() for pattern in patterns.split(",") if pattern.strip()]
    return [pattern.strip() for pattern in patterns if isinstance(pattern, str) and pattern.strip()]


def _matches_any_pattern(name: str, patterns: list[str]) -> bool:
    return not patterns or any(pattern in name for pattern in patterns)


def _tensor_debug_info(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    transfer_ptr, transfer_nbytes = _get_tensor_transfer_region(tensor)
    info: dict[str, Any] = {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "data_ptr": tensor.data_ptr(),
        "transfer_ptr": transfer_ptr,
        "transfer_nbytes": transfer_nbytes,
        "storage_offset": tensor.storage_offset(),
        "is_contiguous": tensor.is_contiguous(),
    }
    try:
        info["storage_nbytes"] = tensor.untyped_storage().nbytes()
    except Exception:
        info["storage_nbytes"] = None

    try:
        if tensor.is_contiguous() or tensor.numel() <= MAX_DEBUG_TENSOR_SAMPLE_ELEMENTS:
            flat_tensor = tensor.detach().view(-1) if tensor.is_contiguous() else tensor.detach().reshape(-1)
            sample_count = min(flat_tensor.numel(), MAX_DEBUG_TENSOR_SAMPLE_ELEMENTS)
            if sample_count > 0:
                if flat_tensor.numel() <= sample_count:
                    sample = flat_tensor
                elif sample_count == 1:
                    sample = flat_tensor[:1]
                else:
                    indices = torch.arange(sample_count, dtype=torch.int64, device=flat_tensor.device)
                    indices = indices * (flat_tensor.numel() - 1) // (sample_count - 1)
                    sample = flat_tensor.index_select(0, indices)
                sample_cpu = sample.detach().cpu().contiguous()
                info["sample_count"] = sample_count
                info["sample_digest"] = hashlib.sha256(sample_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
            else:
                info["sample_count"] = 0
                info["sample_digest"] = None
        else:
            info["sample_count"] = 0
            info["sample_digest"] = None
            info["sample_digest_skipped"] = "non-contiguous large tensor"
    except Exception as e:
        info["sample_count"] = 0
        info["sample_digest"] = None
        info["sample_digest_error"] = str(e)
    return info


def collect_tensor_debug_info(
    model: nn.Module,
    patterns: list[str] | str | None = None,
    limit: int = MAX_DEBUG_TENSOR_VERIFY_COUNT,
) -> dict[str, Any]:
    normalized_patterns = _normalize_debug_patterns(patterns)
    limit = max(0, min(limit, MAX_DEBUG_TENSOR_VERIFY_COUNT))
    selected_tensors = []
    matched_count = 0
    for name, tensor in _iter_transferable_tensors(model):
        if not _matches_any_pattern(name, normalized_patterns):
            continue
        matched_count += 1
        if len(selected_tensors) < limit:
            selected_tensors.append((name, tensor))

    torch.npu.synchronize()
    tensors = {name: _tensor_debug_info(name, tensor) for name, tensor in selected_tensors}
    torch.npu.synchronize()
    return {
        "patterns": normalized_patterns,
        "matched_count": matched_count,
        "returned_count": len(tensors),
        "tensors": tensors,
    }


def _block_contains_weight_ptr(address: int, size: int, sorted_weight_ptrs: list[int]) -> bool:
    index = bisect_left(sorted_weight_ptrs, address)
    return index < len(sorted_weight_ptrs) and sorted_weight_ptrs[index] < address + size


def _iter_transfer_chunks(
    weight_names: list[str],
    seed_ptr_list: list[int],
    client_ptr_list: list[int],
    client_len_list: list[int],
):
    chunk_start = 0
    chunk_bytes = 0
    chunk_weights = 0

    for index, length in enumerate(client_len_list):
        should_flush = chunk_weights > 0 and (
            chunk_bytes + length > MAX_TRANSFER_CHUNK_BYTES or chunk_weights >= MAX_TRANSFER_CHUNK_WEIGHTS
        )
        if should_flush:
            yield (
                weight_names[chunk_start:index],
                seed_ptr_list[chunk_start:index],
                client_ptr_list[chunk_start:index],
                client_len_list[chunk_start:index],
            )
            chunk_start = index
            chunk_bytes = 0
            chunk_weights = 0

        chunk_bytes += length
        chunk_weights += 1

    if chunk_weights > 0:
        yield (
            weight_names[chunk_start:],
            seed_ptr_list[chunk_start:],
            client_ptr_list[chunk_start:],
            client_len_list[chunk_start:],
        )


class RForkTransferBackend:
    def __init__(self):
        self.rfork_transfer_engine: Any | None = None
        self.rfork_transfer_engine_session_id = None
        self.rfork_transfer_engine_weights_info_dict = None
        self.registered_weight_blocks = []
        self._is_initialized = False
        self.init_transfer_engine()

    def init_transfer_engine(self):
        try:
            from yr.datasystem import TransferEngine  # type: ignore[import-not-found]
        except ImportError as e:
            err_msg = (
                "Failed to import TransferEngine from yr.datasystem. "
                "Please install @yuanrong-datasystem/transfer_engine."
            )
            logger.error(err_msg)
            raise ImportError(err_msg) from e

        transfer_engine = TransferEngine()
        local_hostname = join_host_port(get_ip(), get_open_port())
        ret = transfer_engine.initialize(local_hostname, "ascend", f"npu:{torch.npu.current_device()}")
        if ret.is_error():
            err_msg = (
                f"TransferEngine initialization failed: "
                f"initialize({local_hostname}, 'ascend', "
                f"'npu:{int(torch.npu.current_device())}') -> {ret.to_string()}"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        self.rfork_transfer_engine = transfer_engine
        self.rfork_transfer_engine_session_id = local_hostname
        self._is_initialized = True

    def is_initialized(self) -> bool:
        return self._is_initialized

    def _get_transfer_engine(self) -> Any:
        if self.rfork_transfer_engine is None:
            raise RuntimeError("TransferEngine is not initialized.")
        return self.rfork_transfer_engine

    def register_memory_region(self, model):
        transfer_engine = self._get_transfer_engine()
        start_reg_mr_tic = time.time()

        torch.npu.synchronize()
        weight_mr_dict = {}
        weight_addr_set = set()
        for name, weight in _iter_transferable_tensors(model):
            transfer_ptr, transfer_nbytes = _get_tensor_transfer_region(weight)
            weight_mr_dict[name] = (
                transfer_ptr,
                weight.numel(),
                weight.element_size(),
                transfer_nbytes,
            )
            weight_addr_set.add(transfer_ptr)
        sorted_weight_ptrs = sorted(weight_addr_set)

        memory_snapshot = torch.npu.memory.memory_snapshot()
        weight_blocks_for_reg_mr = []
        for segment in memory_snapshot:
            current_weight_block = None
            for block in segment.get("blocks", []):
                address = block.get("address", -1)
                size = block.get("size", -1)
                state = block.get("state", "")
                if address < 0 or size < 0 or state == "":
                    continue
                if state == "active_allocated" and _block_contains_weight_ptr(address, size, sorted_weight_ptrs):
                    if current_weight_block is None:
                        current_weight_block = (address, size)
                    elif current_weight_block[0] + current_weight_block[1] == address:
                        current_weight_block = (
                            current_weight_block[0],
                            current_weight_block[1] + size,
                        )
                    else:
                        weight_blocks_for_reg_mr.append(current_weight_block)
                        current_weight_block = (address, size)
            if current_weight_block is not None:
                weight_blocks_for_reg_mr.append(current_weight_block)

        addresses, sizes = zip(*weight_blocks_for_reg_mr) if weight_blocks_for_reg_mr else ((), ())
        ret = transfer_engine.batch_register_memory(addresses, sizes)
        if ret.is_error():
            logger.error(
                "batch_register_memory failed for %d blocks, ret: %s",
                len(weight_blocks_for_reg_mr),
                ret.to_string(),
            )
            return False

        self.rfork_transfer_engine_weights_info_dict = weight_mr_dict
        self.registered_weight_blocks = weight_blocks_for_reg_mr

        logger.info(
            "register_memory_region time: %.4fs, weights: %d",
            time.time() - start_reg_mr_tic,
            len(weight_mr_dict),
        )
        return True

    def unregister_memory_region(self) -> bool:
        transfer_engine = self._get_transfer_engine()
        start_unreg_mr_tic = time.time()
        if not self.registered_weight_blocks:
            self.rfork_transfer_engine_weights_info_dict = None
            logger.debug("unregister_memory_region skipped because no blocks are registered.")
            return True

        ret = transfer_engine.batch_unregister_memory([address for address, _ in self.registered_weight_blocks])
        if ret.is_error():
            logger.error(
                "batch_unregister_memory failed for %d blocks, ret: %s",
                len(self.registered_weight_blocks),
                ret.to_string(),
            )
            return False
        self.rfork_transfer_engine_weights_info_dict = None
        self.registered_weight_blocks = []
        logger.info(
            "unregister_memory_region time: %.4fs",
            time.time() - start_unreg_mr_tic,
        )
        return True

    def recv_from_source(
        self,
        model,
        seed_instance_ip,
        seed_instance_service_port,
        local_seed_key,
        debug_verify_patterns: list[str] | str | None = None,
    ):
        transfer_engine = self._get_transfer_engine()
        seed_url = f"http://{seed_instance_ip}:{seed_instance_service_port}"
        seed_session_id, seed_weight_info = get_remote_instance_transfer_engine_info(seed_url, local_seed_key)
        if seed_session_id is None or seed_weight_info is None:
            logger.error("Cannot get transfer engine session or weight info.")
            return False

        seed_ptr_list = []
        client_ptr_list = []
        client_len_list = []
        weight_names = []
        for name, tensor in _iter_transferable_tensors(model):
            weight_info = seed_weight_info.get(name, None)
            if weight_info is None:
                logger.error("Cannot find weight info for %s.", name)
                return False

            if len(weight_info) == 3:
                seed_ptr, seed_len, seed_size = weight_info
                seed_nbytes = seed_len * seed_size
            elif len(weight_info) == 4:
                seed_ptr, seed_len, seed_size, seed_nbytes = weight_info
            else:
                logger.error("Invalid weight info for %s: %s", name, weight_info)
                return False

            tensor_ptr, tensor_nbytes = _get_tensor_transfer_region(tensor)
            if seed_len != tensor.numel() or seed_size != tensor.element_size():
                logger.error(
                    "Weight info mismatch for %s, expected (%s, %s), got (%s, %s)",
                    name,
                    seed_len,
                    seed_size,
                    tensor.numel(),
                    tensor.element_size(),
                )
                return False
            if seed_nbytes != tensor_nbytes:
                logger.error(
                    "Weight storage size mismatch for %s, expected %s bytes, got %s bytes",
                    name,
                    seed_nbytes,
                    tensor_nbytes,
                )
                return False

            seed_ptr_list.append(seed_ptr)
            client_ptr_list.append(tensor_ptr)
            client_len_list.append(tensor_nbytes)
            weight_names.append(name)

        start_transfer_tic = time.time()
        transfer_chunks = list(
            _iter_transfer_chunks(
                weight_names,
                seed_ptr_list,
                client_ptr_list,
                client_len_list,
            )
        )
        logger.info(
            "transfer weights starts, weights: %d, chunks: %d, total bytes: %.2f GiB",
            len(client_len_list),
            len(transfer_chunks),
            sum(client_len_list) / (1024**3),
        )
        for index, (chunk_names, chunk_seed_ptrs, chunk_client_ptrs, chunk_lengths) in enumerate(transfer_chunks, 1):
            chunk_start_tic = time.time()
            logger.debug(
                "transfer weights chunk %d/%d starts, weights: %d, bytes: %.2f GiB, first: %s, last: %s",
                index,
                len(transfer_chunks),
                len(chunk_lengths),
                sum(chunk_lengths) / (1024**3),
                chunk_names[0],
                chunk_names[-1],
            )
            ret = transfer_engine.batch_transfer_sync_read(
                seed_session_id,
                chunk_client_ptrs,
                chunk_seed_ptrs,
                chunk_lengths,
            )
            if ret.is_error():
                logger.error(
                    "Failed to transfer weights chunk %d/%d, first: %s, last: %s, ret=%s",
                    index,
                    len(transfer_chunks),
                    chunk_names[0],
                    chunk_names[-1],
                    ret.to_string(),
                )
                return False
            logger.debug(
                "transfer weights chunk %d/%d done, time: %.4fs",
                index,
                len(transfer_chunks),
                time.time() - chunk_start_tic,
            )

        torch.npu.synchronize()
        if _normalize_debug_patterns(debug_verify_patterns):
            if not verify_remote_tensor_debug_info(
                model,
                seed_url,
                local_seed_key,
                debug_verify_patterns,
            ):
                return False
        logger.info("transfer weights time: %.4fs", time.time() - start_transfer_tic)
        return True


def get_remote_instance_transfer_engine_info(seed_url: str, local_seed_key: str):
    try:
        response = requests.get(
            f"{seed_url}/get_rfork_transfer_engine_info",
            params={"seed_key": local_seed_key},
        )
        if response.status_code != 200:
            logger.error(
                "GET %s/get_rfork_transfer_engine_info failed: %s",
                seed_url,
                response.status_code,
            )
            return None, None

        data = response.json()
        info = data.get("rfork_transfer_engine_info", None)
        if info is not None and isinstance(info, list) and len(info) == 2:
            return info[0], info[1]

        logger.error(
            "Failed to get rfork_transfer_engine_info in response from %s.",
            seed_url,
        )
        return None, None
    except Exception as e:
        logger.error("Exception getting transfer engine info from %s: %s", seed_url, e)
        return None, None


def get_remote_tensor_debug_info(
    seed_url: str,
    local_seed_key: str,
    patterns: list[str] | str,
    limit: int = MAX_DEBUG_TENSOR_VERIFY_COUNT,
):
    try:
        response = requests.get(
            f"{seed_url}/debug/rfork_tensor_info",
            params={
                "seed_key": local_seed_key,
                "patterns": ",".join(_normalize_debug_patterns(patterns)),
                "limit": limit,
            },
            timeout=30,
        )
        if response.status_code != 200:
            logger.error(
                "GET %s/debug/rfork_tensor_info failed: %s",
                seed_url,
                response.status_code,
            )
            return None

        data = response.json()
        return data.get("rfork_tensor_info", None)
    except Exception as e:
        logger.error("Exception getting tensor debug info from %s: %s", seed_url, e)
        return None


def verify_remote_tensor_debug_info(
    model: nn.Module,
    seed_url: str,
    local_seed_key: str,
    patterns: list[str] | str,
) -> bool:
    remote_info = get_remote_tensor_debug_info(seed_url, local_seed_key, patterns)
    if remote_info is None:
        logger.error("RFork tensor debug verify failed: cannot fetch seed tensor info.")
        return False

    local_info = collect_tensor_debug_info(model, patterns)
    remote_tensors = remote_info.get("tensors", {})
    local_tensors = local_info.get("tensors", {})
    mismatch_details = []

    for name, local_tensor_info in local_tensors.items():
        remote_tensor_info = remote_tensors.get(name)
        if remote_tensor_info is None:
            mismatch_details.append(f"{name}: missing on seed")
            continue

        for field in ("dtype", "shape", "numel", "element_size", "transfer_nbytes"):
            if local_tensor_info.get(field) != remote_tensor_info.get(field):
                mismatch_details.append(
                    f"{name}: {field} local={local_tensor_info.get(field)} seed={remote_tensor_info.get(field)}"
                )

        local_digest = local_tensor_info.get("sample_digest")
        remote_digest = remote_tensor_info.get("sample_digest")
        if local_digest is not None and remote_digest is not None and local_digest != remote_digest:
            mismatch_details.append(f"{name}: sample_digest local={local_digest} seed={remote_digest}")

    for name in remote_tensors:
        if name not in local_tensors:
            mismatch_details.append(f"{name}: missing on receiver")

    if mismatch_details:
        logger.error(
            "RFork tensor debug verify failed, local matched/returned=%s/%s, "
            "seed matched/returned=%s/%s, mismatches: %s",
            local_info.get("matched_count"),
            local_info.get("returned_count"),
            remote_info.get("matched_count"),
            remote_info.get("returned_count"),
            mismatch_details[:8],
        )
        return False

    logger.info(
        "RFork tensor debug verify passed, patterns=%s, tensors=%d/%d",
        _normalize_debug_patterns(patterns),
        local_info.get("returned_count"),
        local_info.get("matched_count"),
    )
    return True
