# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.utils.yuanrong_transfer_engine import (
    global_yuanrong_te,
)


class YuanrongConnectorWorker(MooncakeConnectorWorker):
    transfer_engine_global = global_yuanrong_te


class YuanrongConnector(MooncakeConnector):
    metadata_cls = MooncakeConnectorMetadata
    worker_cls = YuanrongConnectorWorker
