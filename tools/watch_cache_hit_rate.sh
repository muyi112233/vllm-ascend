#!/usr/bin/env bash

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

set -euo pipefail

METRICS_URL="${METRICS_URL:-http://127.0.0.1:1025/metrics}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-5}"
OUTPUT_FILE=""

usage() {
    cat <<'EOF'
Watch prefix cache hit rate from a Prometheus metrics endpoint.

Usage:
  tools/watch_prefix_cache_hit_rate.sh [OPTIONS]

Options:
  -u, --url URL         Metrics endpoint URL.
                        Default: http://127.0.0.1:1025/metrics
  -i, --interval SEC    Poll interval in seconds.
                        Default: 5
  -o, --output FILE     Append samples to a log file.
  -h, --help            Show this help message.

Environment variables:
  METRICS_URL           Same as --url
  INTERVAL_SECONDS      Same as --interval

Notes:
  - The script reports one row per engine plus one aggregated "all" row.
  - It reports local prefix cache, external prefix cache, and combined hit rates.
  - "ext_total" means external hit rate on local prefix-cache misses.
  - "eff_total" means end-to-end effective hit rate over all prefix-cache queries.
  - "window_hit_rate" is computed from the delta between two polls.
  - "total_hit_rate" is computed from the cumulative counters.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--url)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            METRICS_URL="$2"
            shift 2
            ;;
        -i|--interval)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            INTERVAL_SECONDS="$2"
            shift 2
            ;;
        -o|--output)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! [[ "$INTERVAL_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid interval: $INTERVAL_SECONDS" >&2
    exit 1
fi

fetch_counters() {
    local metrics

    metrics="$(curl -fsS "$METRICS_URL")"
    printf '%s\n' "$metrics" | awk '
        function extract_engine(metric_token,   engine_field) {
            if (match(metric_token, /engine="[^"]+"/) == 0) {
                return "unknown"
            }
            engine_field = substr(metric_token, RSTART, RLENGTH)
            sub(/^engine="/, "", engine_field)
            sub(/"$/, "", engine_field)
            return engine_field
        }

        $1 !~ /^#/ {
            metric_name = $1
            sub(/\{.*/, "", metric_name)
            engine = extract_engine($1)

            if (metric_name == "vllm:prefix_cache_queries_total") {
                local_q[engine] += $NF
                seen[engine] = 1
            } else if (metric_name == "vllm:prefix_cache_hits_total") {
                local_h[engine] += $NF
                seen[engine] = 1
            } else if (metric_name == "vllm:external_prefix_cache_queries_total") {
                ext_q[engine] += $NF
                seen[engine] = 1
            } else if (metric_name == "vllm:external_prefix_cache_hits_total") {
                ext_h[engine] += $NF
                seen[engine] = 1
            }
        }

        END {
            for (engine in seen) {
                printf "%s %.0f %.0f %.0f %.0f\n",
                    engine, local_q[engine], local_h[engine], ext_q[engine], ext_h[engine]
            }
        }
    ' | sort -V -k1,1
}

format_rate() {
    local numerator="$1"
    local denominator="$2"

    awk -v numerator="$numerator" -v denominator="$denominator" '
        BEGIN {
            if (denominator <= 0) {
                printf "n/a"
            } else {
                printf "%.2f%%", (numerator / denominator) * 100
            }
        }
    '
}

print_header() {
    printf '%-19s %-8s %-10s %-10s %-11s %-11s %-10s %-10s %-11s %-11s %-11s\n' \
        "timestamp" \
        "engine" \
        "local_q" \
        "local_h" \
        "local_win" \
        "local_total" \
        "ext_q" \
        "ext_h" \
        "ext_win" \
        "ext_total" \
        "eff_total"
}

cleanup() {
    echo
    echo "Stopped."
}

trap cleanup INT TERM

declare -A previous_local_queries=()
declare -A previous_local_hits=()
declare -A previous_external_queries=()
declare -A previous_external_hits=()

previous_total_local_queries=""
previous_total_local_hits=""
previous_total_external_queries=""
previous_total_external_hits=""

compute_window_rate() {
    local current_queries="$1"
    local current_hits="$2"
    local previous_queries="${3:-}"
    local previous_hits="${4:-}"

    if [[ -z "$previous_queries" || -z "$previous_hits" ]]; then
        printf 'n/a'
        return
    fi

    local delta_queries=$((current_queries - previous_queries))
    local delta_hits=$((current_hits - previous_hits))

    if (( delta_queries < 0 || delta_hits < 0 )); then
        printf 'reset'
        return
    fi

    format_rate "$delta_hits" "$delta_queries"
}

emit_line() {
    local line="$1"
    printf '%s\n' "$line"
    if [[ -n "$OUTPUT_FILE" ]]; then
        printf '%s\n' "$line" >>"$OUTPUT_FILE"
    fi
}

print_header

while true; do
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

    if ! counters="$(fetch_counters 2>/dev/null)"; then
        printf '%-19s %s\n' "$timestamp" "failed to fetch metrics from $METRICS_URL" >&2
        sleep "$INTERVAL_SECONDS"
        continue
    fi

    if [[ -z "$counters" ]]; then
        printf '%-19s %s\n' "$timestamp" "no prefix cache metrics found at $METRICS_URL" >&2
        sleep "$INTERVAL_SECONDS"
        continue
    fi

    declare -A current_local_queries=()
    declare -A current_local_hits=()
    declare -A current_external_queries=()
    declare -A current_external_hits=()

    total_local_queries=0
    total_local_hits=0
    total_external_queries=0
    total_external_hits=0

    while read -r engine local_queries local_hits external_queries external_hits; do
        [[ -n "$engine" ]] || continue

        current_local_queries["$engine"]="$local_queries"
        current_local_hits["$engine"]="$local_hits"
        current_external_queries["$engine"]="$external_queries"
        current_external_hits["$engine"]="$external_hits"

        total_local_queries=$((total_local_queries + local_queries))
        total_local_hits=$((total_local_hits + local_hits))
        total_external_queries=$((total_external_queries + external_queries))
        total_external_hits=$((total_external_hits + external_hits))

        local_window_rate="$(
            compute_window_rate \
                "$local_queries" \
                "$local_hits" \
                "${previous_local_queries[$engine]-}" \
                "${previous_local_hits[$engine]-}"
        )"
        external_window_rate="$(
            compute_window_rate \
                "$external_queries" \
                "$external_hits" \
                "${previous_external_queries[$engine]-}" \
                "${previous_external_hits[$engine]-}"
        )"

        local_total_rate="$(format_rate "$local_hits" "$local_queries")"
        external_total_rate="$(format_rate "$external_hits" "$external_queries")"
        effective_total_rate="$(
            format_rate \
                "$((local_hits + external_hits))" \
                "$local_queries"
        )"

        line="$(printf '%-19s %-8s %-10s %-10s %-11s %-11s %-10s %-10s %-11s %-11s %-11s' \
            "$timestamp" \
            "$engine" \
            "$local_queries" \
            "$local_hits" \
            "$local_window_rate" \
            "$local_total_rate" \
            "$external_queries" \
            "$external_hits" \
            "$external_window_rate" \
            "$external_total_rate" \
            "$effective_total_rate")"
        emit_line "$line"
    done <<<"$counters"

    local_total_rate="$(format_rate "$total_local_hits" "$total_local_queries")"
    external_total_rate="$(format_rate "$total_external_hits" "$total_external_queries")"
    effective_total_rate="$(
        format_rate \
            "$((total_local_hits + total_external_hits))" \
            "$total_local_queries"
    )"
    local_window_rate="$(
        compute_window_rate \
            "$total_local_queries" \
            "$total_local_hits" \
            "$previous_total_local_queries" \
            "$previous_total_local_hits"
    )"
    external_window_rate="$(
        compute_window_rate \
            "$total_external_queries" \
            "$total_external_hits" \
            "$previous_total_external_queries" \
            "$previous_total_external_hits"
    )"

    line="$(printf '%-19s %-8s %-10s %-10s %-11s %-11s %-10s %-10s %-11s %-11s %-11s' \
        "$timestamp" \
        "all" \
        "$total_local_queries" \
        "$total_local_hits" \
        "$local_window_rate" \
        "$local_total_rate" \
        "$total_external_queries" \
        "$total_external_hits" \
        "$external_window_rate" \
        "$external_total_rate" \
        "$effective_total_rate")"
    emit_line "$line"
    printf '\n'
    if [[ -n "$OUTPUT_FILE" ]]; then
        printf '\n' >>"$OUTPUT_FILE"
    fi

    previous_local_queries=()
    previous_local_hits=()
    previous_external_queries=()
    previous_external_hits=()

    while read -r engine local_queries local_hits external_queries external_hits; do
        [[ -n "$engine" ]] || continue
        previous_local_queries["$engine"]="$local_queries"
        previous_local_hits["$engine"]="$local_hits"
        previous_external_queries["$engine"]="$external_queries"
        previous_external_hits["$engine"]="$external_hits"
    done <<<"$counters"

    previous_total_local_queries="$total_local_queries"
    previous_total_local_hits="$total_local_hits"
    previous_total_external_queries="$total_external_queries"
    previous_total_external_hits="$total_external_hits"
    sleep "$INTERVAL_SECONDS"
done
