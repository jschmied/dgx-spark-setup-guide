#!/usr/bin/env bash
# Make the SWE-bench toolchain pick the Docker image architecture from the host
# instead of the hardcoded `x86_64`. Required on aarch64 (DGX Spark / GX10):
# SWE-bench ships arm64 images under docker.io/swebench/sweb.eval.arm64.*, but
# both the mini-swe-agent runner and the official swebench harness default to
# x86_64, which exits instantly with an exec-format error on this CPU.
#
# These edits live INSIDE the respective venvs, so they are LOST on every
# `pipx upgrade mini-swe-agent` / `pip install -U swebench`. Re-run this script
# afterwards. It is idempotent (running twice is a no-op).
#
# Run as the user that owns the installs (here: llm):
#   ./patch-arm64.sh
#
# Override paths via env if your layout differs:
#   MSWEA_VENV_PY=/path/to/mini-swe-agent/venv/bin/python
#   SB_VENV_PY=/path/to/sb-venv/bin/python
set -uo pipefail

MSWEA_VENV_PY="${MSWEA_VENV_PY:-$HOME/.local/share/pipx/venvs/mini-swe-agent/bin/python}"
SB_VENV_PY="${SB_VENV_PY:-$HOME/sb-venv/bin/python}"
rc=0

# --- 1. mini-swe-agent runner: minisweagent/run/benchmarks/swebench.py ---------
if [ -x "$MSWEA_VENV_PY" ]; then
  TARGET="$("$MSWEA_VENV_PY" -c 'import minisweagent, pathlib; print(pathlib.Path(minisweagent.__file__).parent / "run/benchmarks/swebench.py")' 2>/dev/null | tail -1)"
  if [ -f "$TARGET" ]; then
    "$MSWEA_VENV_PY" - "$TARGET" <<'PY' || rc=1
import pathlib, sys
p = pathlib.Path(sys.argv[1]); src = p.read_text()
old = '        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()'
new = ('        arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"\n'
       '        image_name = f"docker.io/swebench/sweb.eval.{arch}.{id_docker_compatible}:latest".lower()')
if "sweb.eval.{arch}." in src:
    print("[mini-swe-agent] already patched"); sys.exit(0)
if "import platform" not in src:
    src = src.replace("import json\n", "import json\nimport platform\n", 1)
if old not in src:
    print("[mini-swe-agent] ERROR: x86_64 image line not found (upstream changed?)", file=sys.stderr); sys.exit(2)
p.write_text(src.replace(old, new, 1)); print("[mini-swe-agent] patched (arm64-aware)")
PY
  else
    echo "[mini-swe-agent] swebench.py not found — skipping"
  fi
else
  echo "[mini-swe-agent] venv python not found at $MSWEA_VENV_PY — skipping"
fi

# --- 2. swebench harness: swebench/harness/run_evaluation.py --------------------
if [ -x "$SB_VENV_PY" ]; then
  TARGET="$("$SB_VENV_PY" -c 'import swebench.harness.run_evaluation as m; print(m.__file__)' 2>/dev/null | tail -1)"
  if [ -f "$TARGET" ]; then
    "$SB_VENV_PY" - "$TARGET" <<'PY' || rc=1
import pathlib, sys
p = pathlib.Path(sys.argv[1]); src = p.read_text()
old = ("            lambda instance: make_test_spec(\n"
       "                instance,\n"
       "                namespace=namespace,\n"
       "                instance_image_tag=instance_image_tag,\n"
       "                env_image_tag=env_image_tag,\n"
       "            ),")
new = ("            lambda instance: make_test_spec(\n"
       "                instance,\n"
       "                namespace=namespace,\n"
       "                instance_image_tag=instance_image_tag,\n"
       "                env_image_tag=env_image_tag,\n"
       '                arch=("arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"),\n'
       "            ),")
if "arch=(" in src and "platform.machine" in src:
    print("[swebench harness] already patched"); sys.exit(0)
if "import platform" not in src:
    src = src.replace("import docker\n", "import docker\nimport platform\n", 1)
if old not in src:
    print("[swebench harness] ERROR: make_test_spec call not found (upstream changed?)", file=sys.stderr); sys.exit(2)
p.write_text(src.replace(old, new, 1)); print("[swebench harness] patched (arm64-aware)")
PY
  else
    echo "[swebench harness] run_evaluation.py not found — skipping"
  fi
else
  echo "[swebench harness] venv python not found at $SB_VENV_PY — skipping (only needed for local scoring)"
fi

exit $rc
