#!/usr/bin/env bash
# Queries OpenShift Prometheus for per-benchmark disk and RDMA metrics on a node.
#
# Usage:
#   ./benchmark-io-metrics.sh <pipelinerun-name> [node] [namespace]
#   ./benchmark-io-metrics.sh --range <start_epoch> <end_epoch> [node]
#
# Examples:
#   ./benchmark-io-metrics.sh mooncake-offloading-208d51
#   ./benchmark-io-metrics.sh --range 1722200000 1722203600 gjfjh
#
# Requires: kubectl access to the diadochos cluster (set KUBECONFIG accordingly),
#           and access to the openshift-monitoring namespace.

set -euo pipefail

PROM_POD="prometheus-k8s-0"
PROM_NS="openshift-monitoring"
DEFAULT_NODE="diadochos-hqxzk-gpu-h100-gjfjh"
DEFAULT_NS="benchflow"
DISK_DEVICE="md0"

prom_query() {
    local query="$1"
    KUBECONFIG="${KUBECONFIG}" kubectl exec -n "$PROM_NS" "$PROM_POD" -c prometheus -- \
        wget -qO- "http://localhost:9090/api/v1/query?query=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$query'''))")" 2>/dev/null
}

prom_query_range() {
    local query="$1" start="$2" end="$3" step="${4:-15}"
    local encoded
    encoded=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$query'''))")
    KUBECONFIG="${KUBECONFIG}" kubectl exec -n "$PROM_NS" "$PROM_POD" -c prometheus -- \
        wget -qO- "http://localhost:9090/api/v1/query_range?query=${encoded}&start=${start}&end=${end}&step=${step}" 2>/dev/null
}

extract_scalar() {
    python3 -c "
import sys, json
d = json.load(sys.stdin)
if d['status'] != 'success':
    print('error', file=sys.stderr); sys.exit(1)
results = d['data']['result']
if not results:
    print('N/A')
else:
    print(results[0]['value'][1])
"
}

extract_range_stats() {
    python3 -c "
import sys, json
d = json.load(sys.stdin)
if d['status'] != 'success':
    print('error'); sys.exit(1)
results = d['data']['result']
if not results:
    print('N/A|N/A|N/A')
else:
    values = [float(v[1]) for v in results[0]['values'] if v[1] != 'NaN']
    if not values:
        print('N/A|N/A|N/A')
    else:
        avg = sum(values) / len(values)
        mx = max(values)
        mn = min(values)
        print(f'{avg}|{mx}|{mn}')
"
}

fmt_bytes() {
    python3 -c "
v = '$1'
if v == 'N/A':
    print('N/A')
else:
    b = float(v)
    if b >= 1e9: print(f'{b/1e9:.2f} GB/s')
    elif b >= 1e6: print(f'{b/1e6:.2f} MB/s')
    elif b >= 1e3: print(f'{b/1e3:.2f} KB/s')
    else: print(f'{b:.2f} B/s')
"
}

fmt_iops() {
    python3 -c "
v = '$1'
if v == 'N/A':
    print('N/A')
else:
    print(f'{float(v):.1f} IOPS')
"
}

fmt_triple() {
    local triple="$1" unit="$2"
    python3 -c "
parts = '$triple'.split('|')
unit = '$unit'
def fmt(v, u):
    if v == 'N/A': return 'N/A'
    f = float(v)
    if u == 'bytes':
        if f >= 1e9: return f'{f/1e9:.2f} GB/s'
        elif f >= 1e6: return f'{f/1e6:.2f} MB/s'
        elif f >= 1e3: return f'{f/1e3:.2f} KB/s'
        else: return f'{f:.2f} B/s'
    else:
        return f'{f:.1f} IOPS'
print(f'avg={fmt(parts[0], u\"$unit\")}  peak={fmt(parts[1], u\"$unit\")}  min={fmt(parts[2], u\"$unit\")}')
"
}

if [[ "${1:-}" == "--range" ]]; then
    START_EPOCH="${2:?start_epoch required}"
    END_EPOCH="${3:?end_epoch required}"
    NODE="${4:-$DEFAULT_NODE}"
else
    PIPELINE_RUN="${1:?pipeline-run name required}"
    NODE="${2:-$DEFAULT_NODE}"
    BFLOW_NS="${3:-$DEFAULT_NS}"
    HUB_KUBECONFIG="${HUB_KUBECONFIG:-$KUBECONFIG}"

    echo "Looking up PipelineRun $PIPELINE_RUN in $BFLOW_NS..."

    times=$(KUBECONFIG="${HUB_KUBECONFIG}" kubectl get pipelinerun -n "$BFLOW_NS" "$PIPELINE_RUN" \
        -o jsonpath='{.status.startTime} {.status.completionTime}' 2>/dev/null || true)

    if [[ -z "$times" ]]; then
        echo "ERROR: PipelineRun '$PIPELINE_RUN' not found in namespace '$BFLOW_NS'"
        echo "Use --range <start_epoch> <end_epoch> instead"
        exit 1
    fi

    start_time=$(echo "$times" | awk '{print $1}')
    end_time=$(echo "$times" | awk '{print $2}')

    if [[ -z "$end_time" || "$end_time" == "null" ]]; then
        end_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        echo "PipelineRun still running, using now as end time"
    fi

    START_EPOCH=$(python3 -c "from datetime import datetime,timezone; print(int(datetime.strptime('$start_time','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()))")
    END_EPOCH=$(python3 -c "from datetime import datetime,timezone; print(int(datetime.strptime('$end_time','%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()))")

    echo "Time window: $start_time → $end_time"
fi

DURATION=$((END_EPOCH - START_EPOCH))
echo "Node: $NODE"
echo "Duration: ${DURATION}s ($((DURATION / 60))m)"
echo ""

STEP=15
if (( DURATION > 7200 )); then STEP=60; fi
RANGE="${DURATION}s"
if (( DURATION < 120 )); then RANGE="2m"; fi

echo "=== Disk I/O (${DISK_DEVICE}) ==="
echo ""

echo "Write bandwidth:"
stats=$(prom_query_range "rate(node_disk_written_bytes_total{instance=~\".*${NODE}.*\",device=\"${DISK_DEVICE}\"}[2m])" "$START_EPOCH" "$END_EPOCH" "$STEP" | extract_range_stats)
fmt_triple "$stats" "bytes"

echo "Write IOPS:"
stats=$(prom_query_range "rate(node_disk_writes_completed_total{instance=~\".*${NODE}.*\",device=\"${DISK_DEVICE}\"}[2m])" "$START_EPOCH" "$END_EPOCH" "$STEP" | extract_range_stats)
fmt_triple "$stats" "iops"

echo "Read bandwidth:"
stats=$(prom_query_range "rate(node_disk_read_bytes_total{instance=~\".*${NODE}.*\",device=\"${DISK_DEVICE}\"}[2m])" "$START_EPOCH" "$END_EPOCH" "$STEP" | extract_range_stats)
fmt_triple "$stats" "bytes"

echo "Read IOPS:"
stats=$(prom_query_range "rate(node_disk_reads_completed_total{instance=~\".*${NODE}.*\",device=\"${DISK_DEVICE}\"}[2m])" "$START_EPOCH" "$END_EPOCH" "$STEP" | extract_range_stats)
fmt_triple "$stats" "iops"

echo ""

echo "Total bytes written during window:"
start_val=$(prom_query "node_disk_written_bytes_total{instance=~\".*${NODE}.*\",device=\"${DISK_DEVICE}\"} @ $START_EPOCH" | extract_scalar)
end_val=$(prom_query "node_disk_written_bytes_total{instance=~\".*${NODE}.*\",device=\"${DISK_DEVICE}\"} @ $END_EPOCH" | extract_scalar)
python3 -c "
s, e = '$start_val', '$end_val'
if 'N/A' in (s, e):
    print('N/A (counter not available for full window)')
else:
    delta = float(e) - float(s)
    if delta >= 1e12: print(f'{delta/1e12:.2f} TB')
    elif delta >= 1e9: print(f'{delta/1e9:.2f} GB')
    elif delta >= 1e6: print(f'{delta/1e6:.2f} MB')
    else: print(f'{delta:.0f} bytes')
"

echo ""
echo "=== RDMA / InfiniBand ==="
echo ""

for dev in mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7; do
    rcv_stats=$(prom_query_range "rate(node_infiniband_port_data_received_bytes_total{instance=~\".*${NODE}.*\",device=\"${dev}\"}[2m])" "$START_EPOCH" "$END_EPOCH" "$STEP" | extract_range_stats)
    xmt_stats=$(prom_query_range "rate(node_infiniband_port_data_transmitted_bytes_total{instance=~\".*${NODE}.*\",device=\"${dev}\"}[2m])" "$START_EPOCH" "$END_EPOCH" "$STEP" | extract_range_stats)

    rcv_avg=$(echo "$rcv_stats" | cut -d'|' -f1)
    xmt_avg=$(echo "$xmt_stats" | cut -d'|' -f1)

    if [[ "$rcv_avg" != "N/A" ]] && python3 -c "exit(0 if float('$rcv_avg') > 0 else 1)" 2>/dev/null; then
        echo "${dev} receive:"
        fmt_triple "$rcv_stats" "bytes"
    fi
    if [[ "$xmt_avg" != "N/A" ]] && python3 -c "exit(0 if float('$xmt_avg') > 0 else 1)" 2>/dev/null; then
        echo "${dev} transmit:"
        fmt_triple "$xmt_stats" "bytes"
    fi
done

echo ""
echo "Total RDMA bytes received (mlx5_7) during window:"
start_val=$(prom_query "node_infiniband_port_data_received_bytes_total{instance=~\".*${NODE}.*\",device=\"mlx5_7\"} @ $START_EPOCH" | extract_scalar)
end_val=$(prom_query "node_infiniband_port_data_received_bytes_total{instance=~\".*${NODE}.*\",device=\"mlx5_7\"} @ $END_EPOCH" | extract_scalar)
python3 -c "
s, e = '$start_val', '$end_val'
if 'N/A' in (s, e):
    print('N/A (counter not available for full window)')
else:
    delta = float(e) - float(s)
    if delta >= 1e12: print(f'{delta/1e12:.2f} TB')
    elif delta >= 1e9: print(f'{delta/1e9:.2f} GB')
    elif delta >= 1e6: print(f'{delta/1e6:.2f} MB')
    else: print(f'{delta:.0f} bytes')
"

echo ""
echo "=== Disk I/O per NVMe drive (individual) ==="
echo ""
for dev in nvme1n1 nvme2n1 nvme3n1 nvme4n1 nvme5n1 nvme6n1 nvme7n1; do
    stats=$(prom_query_range "rate(node_disk_written_bytes_total{instance=~\".*${NODE}.*\",device=\"${dev}\"}[2m])" "$START_EPOCH" "$END_EPOCH" "$STEP" | extract_range_stats)
    avg=$(echo "$stats" | cut -d'|' -f1)
    peak=$(echo "$stats" | cut -d'|' -f2)
    if [[ "$avg" != "N/A" ]]; then
        printf "%-10s write avg=%-12s peak=%-12s\n" "$dev" "$(fmt_bytes "$avg")" "$(fmt_bytes "$peak")"
    fi
done
