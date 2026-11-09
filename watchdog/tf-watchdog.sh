#!/usr/bin/env bash
# tf-watchdog.sh — Liveness, Wedged-Engine Detection & Memory Guard Watchdog for 2x DGX Spark TensorFold
#
# Checks:
# 1. HTTP 200 on /v1/models
# 2. 5-token completion probe (detects "wedged" CUDA stream states where /v1/models returns 200 but generation 500s)
# 3. Container liveness on both Rank 0 (Head) and Rank 1 (Worker)
# 4. MemAvailable >= 4.0 GiB on both nodes
# 5. Memguard / OOM traces in container logs over the last 30 minutes
#
# Usage: Run via cron every 15-30 minutes. Silent (exit 0) when healthy; prints details and exits 1 on anomaly.
set -uo pipefail

HEAD_SSH="${HEAD_SSH:-user@192.168.200.16}"
API_URL="${API_URL:-http://192.168.200.16:8888}"
SERVED_MODEL="${SERVED_MODEL:-deepseek-v4-flash-dspark}"
MEM_ALERT_G="${MEM_ALERT_G:-4.0}"
PROBE_OUT="/tmp/tf_probe_last.json"

WEDGED=""
PROBLEMS=""

# 1) API health (/v1/models)
API="ok"
curl -sf --max-time 8 "$API_URL/v1/models" >/dev/null 2>&1 || { API="DOWN"; PROBLEMS="$PROBLEMS API_DOWN"; }

# 1b) Completion probe: catches wedged engine (models=200 but completions=5xx)
PROBE="skip"
if [ "$API" = "ok" ]; then
  PROBE_CODE=$(curl -s -m 45 -o "$PROBE_OUT" -w "%{http_code}" "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":5}" 2>/dev/null || echo 000)
  if [ "$PROBE_CODE" = "200" ]; then
    PROBE="ok"
  elif [ "$PROBE_CODE" = "000" ]; then
    PROBE="TIMEOUT"
    PROBLEMS="$PROBLEMS PROBE_TIMEOUT"
  else
    PROBE="WEDGED(http=$PROBE_CODE)"
    PROBLEMS="$PROBLEMS COMPLETION_WEDGED"
    WEDGED=1
  fi
fi

# 2) Rank 0 container Up (Head node)
UP_HEAD=$(ssh -o GSSAPIAuthentication=no -o ConnectTimeout=6 "$HEAD_SSH" 'docker ps --format "{{.Names}}:{{.Status}}" | grep dsv41-1017-exl3-tf' 2>/dev/null | head -1)
echo "$UP_HEAD" | grep -q "Up" || PROBLEMS="$PROBLEMS RANK0_DOWN"

# 3) Rank 1 container Up (Local Worker node)
UP_WORK=$(docker ps --format '{{.Names}}:{{.Status}}' | grep 'dsv41-1017-exl3-tf' | head -1)
echo "$UP_WORK" | grep -q "Up" || PROBLEMS="$PROBLEMS RANK1_DOWN"

# 4) Served model ID verification
MODEL="ok"
if [ "$API" = "ok" ]; then
  curl -s --max-time 8 "$API_URL/v1/models" | grep -q "$SERVED_MODEL" || { MODEL="WRONG"; PROBLEMS="$PROBLEMS MODEL_WRONG"; }
fi

# 5) Host MemAvailable check on both Sparks
MEM_W=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
MEM_H=$(ssh -o GSSAPIAuthentication=no -o ConnectTimeout=6 "$HEAD_SSH" 'grep MemAvailable /proc/meminfo' 2>/dev/null | awk '{printf "%.1f", $2/1048576}')
[ -n "$MEM_H" ] && awk -v m="$MEM_H" -v a="$MEM_ALERT_G" 'BEGIN{exit !(m+0 < a)}' && PROBLEMS="$PROBLEMS MEM_HEAD_LOW(${MEM_H}G)"
awk -v m="$MEM_W" -v a="$MEM_ALERT_G" 'BEGIN{exit !(m+0 < a)}' && PROBLEMS="$PROBLEMS MEM_WORK_LOW(${MEM_W}G)"

# 6) Memguard / OOM trace check (30m window)
KILLS_W=$(docker logs --since 30m dsv41-1017-exl3-tf 2>/dev/null | grep -ciE "memguard|oom|killed" || true)
KILLS_H=$(ssh -o GSSAPIAuthentication=no -o ConnectTimeout=6 "$HEAD_SSH" 'docker logs --since 30m dsv41-1017-exl3-tf 2>/dev/null | grep -ciE "memguard|oom|killed" || true' 2>/dev/null || echo 0)
KILLS=$(( ${KILLS_W:-0} + ${KILLS_H:-0} ))
[ "${KILLS:-0}" -gt 0 ] 2>/dev/null && PROBLEMS="$PROBLEMS MEMGUARD_TRACE(${KILLS})"

# 7) Auto-restart wedged engine via systemd on Head node
RESTART_NOTE=""
if [ -n "$WEDGED" ]; then
  USTATE=$(ssh -o GSSAPIAuthentication=no -o ConnectTimeout=6 "$HEAD_SSH" 'systemctl --user is-active dsv41-tensorfold.service' 2>/dev/null || echo unknown)
  if [ "$USTATE" = "active" ]; then
    (sleep 2; ssh -o GSSAPIAuthentication=no -o ConnectTimeout=10 "$HEAD_SSH" 'systemctl --user restart dsv41-tensorfold.service') >/dev/null 2>&1 &
    RESTART_NOTE="AUTO_RESTART_ISSUED(head)"
    echo "$(date '+%F %T') wedged($PROBE) -> auto-restart dsv41-tensorfold on head" >> /tmp/tf-watchdog-restarts.log
  else
    RESTART_NOTE="unit_state=$USTATE(handled_by_systemd)"
  fi
fi

if [ -z "$PROBLEMS" ]; then
  exit 0
fi
echo "ALERT: DSV4.1-TF(E1-2.50bpw) Anomaly:$PROBLEMS"
echo "API=$API PROBE=$PROBE RANK0=$UP_HEAD RANK1=$UP_WORK MODEL=$MODEL MEM_HEAD=${MEM_H}G MEM_WORK=${MEM_W}G $RESTART_NOTE"
exit 1
