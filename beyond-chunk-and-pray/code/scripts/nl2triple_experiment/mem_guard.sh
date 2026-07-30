#!/usr/bin/env bash
# Run a command under a memory watchdog so it can never freeze the node.
#
# Launches "$@" in its own process group, then polls /proc/meminfo. If system-wide
# MemAvailable drops below FLOOR_GIB, it SIGKILLs the whole group and exits 137 —
# the box keeps enough headroom to stay responsive, and unlike a silent unified-memory
# freeze this leaves a log line saying exactly what happened. The child's stdout+stderr
# go to STAGE_LOG.
set -u

FLOOR_GIB="${FLOOR_GIB:-12}"
POLL_SEC="${POLL_SEC:-2}"
FLOOR_KB=$((FLOOR_GIB * 1024 * 1024))
STAGE_LOG="${STAGE_LOG:?set STAGE_LOG to the child log path}"

avail_mib() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

setsid bash -c 'exec "$@"' _ "$@" >"$STAGE_LOG" 2>&1 &
PID=$!
# setsid makes the child its own session/group leader, so its PGID == its PID.
PGID=$(ps -o pgid= "$PID" 2>/dev/null | tr -d ' '); PGID=${PGID:-$PID}
echo "[guard] pid=$PID pgid=$PGID floor=${FLOOR_GIB}GiB poll=${POLL_SEC}s avail=$(avail_mib)MiB log=$STAGE_LOG"
echo "[guard] cmd: $*"

while kill -0 "$PID" 2>/dev/null; do
  A=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "${A:-0}" -lt "$FLOOR_KB" ]; then
    echo "[guard] MemAvailable=$((A/1024))MiB < ${FLOOR_GIB}GiB floor -> SIGKILL pgid=$PGID (protecting node)"
    kill -KILL -"$PGID" 2>/dev/null
    sleep "$POLL_SEC"
    echo "[guard] killed; partial output in $STAGE_LOG"
    exit 137
  fi
  sleep "$POLL_SEC"
done

wait "$PID"; RC=$?
echo "[guard] child exited rc=$RC avail=$(avail_mib)MiB"
exit "$RC"
