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

avail_kb()  { awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null; }
avail_mib() { local a; a=$(avail_kb); [ -n "$a" ] && echo $((a / 1024)) || echo "?"; }

# Refuse to start if MemAvailable is unreadable (macOS, some containers, a
# hardened runtime). Without this the first poll reads an empty string, the
# "${A:-0}" default makes 0 < FLOOR true, and the guard SIGKILLs the job it was
# meant to protect within POLL_SEC -- logging "MemAvailable=0MiB", which blames
# memory pressure that never happened. Failing loudly here beats a multi-hour
# datagen/train dying on a lie; silently running unguarded is worse still, since
# the whole point is keeping the node responsive.
if [ -z "$(avail_kb)" ]; then
    echo "[guard] FATAL: cannot read MemAvailable from /proc/meminfo." >&2
    echo "[guard] This guard is Linux-only. Run the command directly if the" >&2
    echo "[guard] node does not need protecting, or run it on Linux." >&2
    exit 2
fi

setsid bash -c 'exec "$@"' _ "$@" >"$STAGE_LOG" 2>&1 &
PID=$!
# setsid makes the child its own session/group leader, so its PGID == its PID.
PGID=$(ps -o pgid= "$PID" 2>/dev/null | tr -d ' '); PGID=${PGID:-$PID}
echo "[guard] pid=$PID pgid=$PGID floor=${FLOOR_GIB}GiB poll=${POLL_SEC}s avail=$(avail_mib)MiB log=$STAGE_LOG"
echo "[guard] cmd: $*"

while kill -0 "$PID" 2>/dev/null; do
  A=$(avail_kb)
  # Only ever kill on a reading we actually parsed. A transient read failure
  # mid-run warns and retries rather than executing the child.
  if [ -z "$A" ]; then
    echo "[guard] WARN: MemAvailable unreadable this poll; not killing, retrying"
  elif [ "$A" -lt "$FLOOR_KB" ]; then
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
