#!/bin/bash
set -euo pipefail

# Standalone Yuanrong worker startup script for validating SSD spill behavior.
# Usage:
#   bash run_yr_worker_spill_ssd.sh [HOST_IP] [WORKER_PORT] [ETCD_PORT]

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

ulimit -l unlimited || echo "Warning: failed to set ulimit -l unlimited"

HOST_IP="${1:-${HOST_IP:-100.100.135.173}}"
WORKER_PORT="${2:-${WORKER_PORT:-18481}}"
ETCD_PORT="${3:-${ETCD_PORT:-19099}}"
ETCD_PEER_PORT="${ETCD_PEER_PORT:-$((ETCD_PORT + 1))}"

ETCD_BIN="${ETCD_BIN:-./etcd}"
ETCDCTL_BIN="${ETCDCTL_BIN:-etcdctl}"
ETCD_DATA_DIR="${ETCD_DATA_DIR:-/tmp/etcd-yuanrong}"

SHM_SIZE_MB="${SHM_SIZE_MB:-512000}"
NODE_TIMEOUT="${NODE_TIMEOUT:-30}"
NODE_DEAD_TIMEOUT="${NODE_DEAD_TIMEOUT:-60}"
LIVENESS_PATH="${LIVENESS_PATH:-/workspace/liveness}"
ARENA_PER_TENANT="${ARENA_PER_TENANT:-1}"

SPILL_DIR="${SPILL_DIR:-/data/ssd/yr_kv_spill}"
SPILL_SIZE_LIMIT="${SPILL_SIZE_LIMIT:-214748364800}"
SPILL_THREAD_NUM="${SPILL_THREAD_NUM:-8}"
SPILL_FILE_MAX_SIZE_MB="${SPILL_FILE_MAX_SIZE_MB:-200}"
SPILL_FILE_OPEN_LIMIT="${SPILL_FILE_OPEN_LIMIT:-512}"
SPILL_ENABLE_READAHEAD="${SPILL_ENABLE_READAHEAD:-true}"

LOG_MONITOR="${LOG_MONITOR:-true}"
LOG_MONITOR_EXPORTER="${LOG_MONITOR_EXPORTER:-harddisk}"
LOG_MONITOR_INTERVAL_MS="${LOG_MONITOR_INTERVAL_MS:-5000}"

if [ ! -x "${ETCD_BIN}" ]; then
  if command -v etcd >/dev/null 2>&1; then
    ETCD_BIN="$(command -v etcd)"
  else
    echo "Cannot find executable etcd. Put etcd in current directory or set ETCD_BIN." >&2
    exit 1
  fi
fi

if ! command -v "${ETCDCTL_BIN}" >/dev/null 2>&1; then
  echo "Cannot find etcdctl. Put it in PATH or set ETCDCTL_BIN." >&2
  exit 1
fi

echo "Stopping old etcd and datasystem_worker processes"
pkill -9 -f '(^|/)etcd( |$)' || true
pkill -9 -f 'datasystem_worker' || true
sleep 2

rm -rf "${ETCD_DATA_DIR}"
rm -rf ./datasystem
rm -rf ~/.datasystem
mkdir -p "${SPILL_DIR}"

echo "Starting etcd"
echo "  endpoint=http://${HOST_IP}:${ETCD_PORT}"
echo "  peer=http://${HOST_IP}:${ETCD_PEER_PORT}"

"${ETCD_BIN}" \
  --name etcd-yuanrong \
  --data-dir "${ETCD_DATA_DIR}" \
  --listen-client-urls "http://${HOST_IP}:${ETCD_PORT}" \
  --advertise-client-urls "http://${HOST_IP}:${ETCD_PORT}" \
  --listen-peer-urls "http://${HOST_IP}:${ETCD_PEER_PORT}" \
  --initial-advertise-peer-urls "http://${HOST_IP}:${ETCD_PEER_PORT}" \
  --initial-cluster "etcd-yuanrong=http://${HOST_IP}:${ETCD_PEER_PORT}" \
  > /tmp/etcd.log 2>&1 &

TIMEOUT="${ETCD_READY_TIMEOUT:-5}"
INTERVAL="${ETCD_READY_INTERVAL:-0.5}"
max_try=$(awk "BEGIN{print int((${TIMEOUT}+${INTERVAL}-0.001)/${INTERVAL})}")

for ((i = 1; i <= max_try; i++)); do
  if "${ETCDCTL_BIN}" --endpoints="http://${HOST_IP}:${ETCD_PORT}" --command-timeout=1s \
       endpoint health --cluster 2>/dev/null | grep -q 'is healthy'; then
    echo "etcd is healthy"
    break
  fi
  sleep "${INTERVAL}"
done

if (( i > max_try )); then
  echo "etcd health check failed after ${TIMEOUT}s. Check /tmp/etcd.log." >&2
  exit 1
fi

echo "Starting Datasystem worker for SSD spill test"
echo "  worker_address=${HOST_IP}:${WORKER_PORT}"
echo "  etcd_address=${HOST_IP}:${ETCD_PORT}"
echo "  shared_memory_size_mb=${SHM_SIZE_MB}"
echo "  spill_directory=${SPILL_DIR}"
echo "  spill_size_limit=${SPILL_SIZE_LIMIT}"

dscli start \
  --timeout 600 \
  -w \
  --worker_address "${HOST_IP}:${WORKER_PORT}" \
  --etcd_address "${HOST_IP}:${ETCD_PORT}" \
  --shared_memory_size_mb "${SHM_SIZE_MB}" \
  --node_timeout_s "${NODE_TIMEOUT}" \
  --node_dead_timeout_s "${NODE_DEAD_TIMEOUT}" \
  --liveness_check_path "${LIVENESS_PATH}" \
  --arena_per_tenant "${ARENA_PER_TENANT}" \
  --spill_directory "${SPILL_DIR}" \
  --spill_size_limit "${SPILL_SIZE_LIMIT}" \
  --spill_thread_num "${SPILL_THREAD_NUM}" \
  --spill_file_max_size_mb "${SPILL_FILE_MAX_SIZE_MB}" \
  --spill_file_open_limit "${SPILL_FILE_OPEN_LIMIT}" \
  --spill_enable_readahead "${SPILL_ENABLE_READAHEAD}" \
  --log_monitor "${LOG_MONITOR}" \
  --log_monitor_exporter "${LOG_MONITOR_EXPORTER}" \
  --log_monitor_interval_ms "${LOG_MONITOR_INTERVAL_MS}"
