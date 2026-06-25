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

import threading
import time
from dataclasses import dataclass

from vllm.logger import logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.model_loader.rfork.seed_protocol import RForkSeedProtocol
from vllm_ascend.model_loader.rfork.seed_server import start_rfork_server
from vllm_ascend.model_loader.rfork.transfer_backend import (
    RForkTransferBackend,
)

SEED_RETRY_INTERVAL_SEC = 1.0


@dataclass(frozen=True)
class RForkFallbackRaceResult:
    seed_available: bool
    fallback_model_path: str | None = None


class RForkWorker:
    def __init__(
        self,
        disaggregation_mode: str,
        node_rank: int,
        tp_rank: int,
        device_id: int,
        scheduler_url: str,
        model_url: str,
        model_deploy_strategy_name: str,
        seed_timeout_sec: float = 30.0,
        seed_key_separator: str = "$",
        is_draft_model: bool = False,
        pp_rank: int | None = None,
        ep_rank: int | None = None,
    ):
        self.device_id = device_id
        self.rfork_seed = None
        self.transfer_backend = RForkTransferBackend()
        self.ready_to_start_seed_service = False
        self.seed_service_started = False
        self.seed_timeout_sec = seed_timeout_sec
        self.seed_protocol = RForkSeedProtocol(
            disaggregation_mode=disaggregation_mode,
            node_rank=node_rank,
            tp_rank=tp_rank,
            scheduler_url=scheduler_url,
            model_url=model_url,
            model_deploy_strategy_name=model_deploy_strategy_name,
            seed_key_separator=seed_key_separator,
            is_draft_worker=is_draft_model,
            pp_rank=pp_rank,
            ep_rank=ep_rank,
        )
        self.fallback_model_path: str | None = None
        self._fallback_path_thread: threading.Thread | None = None
        self._fallback_path_error: Exception | None = None
        self._fallback_path_lock = threading.Lock()

    def is_seed_available(self) -> bool:
        self.rfork_seed = self.seed_protocol.get_seed()
        return self.rfork_seed is not None

    def pre_transfer(self, model) -> bool:
        try:
            assert self.transfer_backend.is_initialized(), "transfer_backend is not initialized, cannot pre_transfer."
            result = self.transfer_backend.register_memory_region(model)
            self.ready_to_start_seed_service = result
            return result
        except AssertionError as e:
            logger.exception("Pre-transfer failed for device_id=%s: %s", self.device_id, e)
            return False

    def reset_transfer_state(self) -> None:
        try:
            self.transfer_backend.unregister_memory_region()
        except Exception as e:
            logger.warning("Failed to unregister rfork memory region: %s", e)
        self.ready_to_start_seed_service = False

    def transfer(self, model) -> bool:
        try:
            assert self.transfer_backend.is_initialized(), "transfer_backend is not initialized, cannot transfer."
            assert self.rfork_seed is not None, "rfork seed is None, cannot transfer."
            return self.transfer_backend.recv_from_source(
                model=model,
                seed_instance_ip=self.rfork_seed["seed_ip"],
                seed_instance_service_port=self.rfork_seed["seed_port"],
                local_seed_key=self.seed_protocol.get_local_seed_key(),
            )
        except AssertionError as e:
            logger.exception(
                "Transfer failed for device_id=%s: %s",
                self.device_id,
                e,
            )
            return False

    def post_transfer(self):
        if self.rfork_seed is None:
            logger.info("rfork seed is None, no need to release.")
            return True
        self.seed_protocol.release_seed(self.rfork_seed)
        self.rfork_seed = None
        return True

    def start_seed_service(self, model):
        if self.seed_service_started:
            logger.info("Seed service already started, skipping.")
            return

        if not self.ready_to_start_seed_service:
            if not self.pre_transfer(model):
                logger.warning(
                    "start_seed_service aborted for device_id=%s: pre_transfer failed",
                    self.device_id,
                )
                return

        port = start_rfork_server(
            self.seed_protocol.get_local_seed_key(),
            (
                self.transfer_backend.rfork_transfer_engine_session_id,
                self.transfer_backend.rfork_transfer_engine_weights_info_dict,
                self.transfer_backend.rfork_transfer_engine_weights_shape_dict,
            ),
            health_timeout_sec=self.seed_timeout_sec,
        )
        if port <= 0:
            logger.warning("start_seed_service failed for device_id=%s", self.device_id)
            return

        self.rfork_heartbeat_thread = threading.Thread(
            target=self.seed_protocol.report_seed,
            args=(port,),
            daemon=True,
            name="RForkHeartbeat",
        )
        self.rfork_heartbeat_thread.start()
        logger.info("Seed service started for device_id=%s, port=%s", self.device_id, port)
        self.seed_service_started = True

    def prefetch_fallback_model_path(self) -> None:
        """Resolve the fallback model path in the background."""
        if not envs_ascend.VLLM_ASCEND_ASYNC_MODEL_MOUNT:
            return

        with self._fallback_path_lock:
            if self.fallback_model_path is not None:
                return
            if self._fallback_path_thread is not None and self._fallback_path_thread.is_alive():
                return

            self._fallback_path_error = None
            self._fallback_path_thread = threading.Thread(
                target=self._resolve_fallback_model_path,
                daemon=True,
                name="RForkFallbackPathResolver",
            )
            self._fallback_path_thread.start()

    def _resolve_fallback_model_path(self) -> None:
        try:
            from vllm_ascend.model_loader.async_mount import get_model_path

            self.fallback_model_path = get_model_path(with_weights=True)
            if self.fallback_model_path:
                logger.info("Resolved RFork fallback model path: %s", self.fallback_model_path)
            else:
                logger.warning("Async model mount did not return a fallback model path.")
        except Exception as exc:
            self._fallback_path_error = exc
            logger.exception("Failed to resolve RFork fallback model path.")

    def get_fallback_model_path(self) -> str | None:
        """Return the fallback model path, waiting for prefetch if needed."""
        if not envs_ascend.VLLM_ASCEND_ASYNC_MODEL_MOUNT:
            return None

        thread = self._fallback_path_thread
        if thread is None:
            self._resolve_fallback_model_path()
        else:
            thread.join()

        if self._fallback_path_error is not None:
            raise RuntimeError("Failed to resolve RFork fallback model path.") from self._fallback_path_error

        return self.fallback_model_path

    def wait_for_seed_or_fallback_model_path(
        self,
        retry_interval_sec: float = SEED_RETRY_INTERVAL_SEC,
    ) -> RForkFallbackRaceResult:
        """Race async mount completion against repeated RFork seed lookup."""
        if not envs_ascend.VLLM_ASCEND_ASYNC_MODEL_MOUNT:
            return RForkFallbackRaceResult(seed_available=False)

        self.prefetch_fallback_model_path()
        logger.info("Initial RFork seed lookup failed; racing seed retry with async model mount fallback.")

        while True:
            if self._fallback_path_error is not None:
                raise RuntimeError("Failed to resolve RFork fallback model path.") from self._fallback_path_error

            self.rfork_seed = self.seed_protocol.get_seed()
            if self.rfork_seed is not None:
                logger.info("RFork seed retry won fallback race.")
                return RForkFallbackRaceResult(seed_available=True)

            if self.fallback_model_path:
                logger.info("Async model mount won RFork fallback race.")
                return RForkFallbackRaceResult(
                    seed_available=False,
                    fallback_model_path=self.fallback_model_path,
                )

            fallback_thread = self._fallback_path_thread
            if fallback_thread is not None and not fallback_thread.is_alive():
                logger.warning(
                    "Async model mount finished without a fallback model path; falling back to original model path."
                )
                return RForkFallbackRaceResult(seed_available=False)

            if fallback_thread is None:
                time.sleep(retry_interval_sec)
            else:
                fallback_thread.join(timeout=retry_interval_sec)
