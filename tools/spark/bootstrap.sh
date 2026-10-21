#!/bin/bash
# Spark (idempotent): create the layout under $R and print the environment facts.
set -u
R=${SLOTSTREAM_R_REMOTE:-$HOME/slotstream-spark}
mkdir -p "$R"/{model/qwen38-flash-next-mlx-4bit,model/packed,pylib,runs,tmp/triton,run/jobs,hf,lb/gguf,state,prefix}

uid=$(id -u)
cg=/sys/fs/cgroup/user.slice/user-$uid.slice/user@$uid.service/cgroup.controllers
if [ -r "$cg" ] && tr ' ' '\n' < "$cg" | grep -qx memory; then memcg=yes; else memcg=no; fi
if [ -r "$cg" ] && tr ' ' '\n' < "$cg" | grep -qx io; then iod=yes; else iod=no; fi
linger=$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || echo unknown)
sysd=$(XDG_RUNTIME_DIR=/run/user/$uid systemctl --user is-system-running 2>/dev/null || true)
dock=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo unavailable)
nvcc_bin=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc 2>/dev/null || true)
if [ -n "$nvcc_bin" ]; then nvcc_v=$("$nvcc_bin" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'); else nvcc_v=missing; fi
py=$(python3 --version 2>&1 | awk '{print $2}')
lbs=$(cat /sys/block/${SLOTSTREAM_NVME_DEVICE:-nvme0n1}/queue/logical_block_size 2>/dev/null || echo unknown)
free_gb=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)

{
  echo "memcg_delegated=$memcg"
  echo "io_delegated=$iod"
  echo "linger=$linger"
  echo "systemd_user=${sysd:-unknown}"
  echo "docker=$dock"
  echo "nvcc=$nvcc_v"
  echo "python3=$py"
  echo "logical_block_size=$lbs"
  echo "free_gb=$free_gb"
} | tee "$R/bootstrap.txt"
