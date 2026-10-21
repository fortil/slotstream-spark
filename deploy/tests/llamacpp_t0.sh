#!/bin/bash
# Spark self-test for deploy/lb/llamacpp.sh (no server is started).
# Prints "llamacpp_t0: <n> passed, <m> failed"; exit 1 on any failure.
cd "$(dirname "$0")/../.." || exit 1
L=deploy/lb/llamacpp.sh
PASS=0 FAIL=0
ok() { PASS=$((PASS+1)); echo "ok   $1"; }
bad() { FAIL=$((FAIL+1)); echo "FAIL $1"; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }
footprint() { bash $L --print-footprint "$@" 2>/dev/null; }

check "q41 A D1"  "[ \"\$(footprint --weights q41 --arm A --mode D1)\" = 'FOOTPRINT_MIB=18944 MEMORYMAX_MIB=10240' ]"
check "q41 A D2"  "[ \"\$(footprint --weights q41 --arm A --mode D2)\" = 'FOOTPRINT_MIB=25088 MEMORYMAX_MIB=16384' ]"
check "q41 A D5"  "[ \"\$(footprint --weights q41 --arm A --mode D5)\" = 'FOOTPRINT_MIB=57344 MEMORYMAX_MIB=47616' ]"
check "q41 B D1"  "[ \"\$(footprint --weights q41 --arm B --mode D1)\" = 'FOOTPRINT_MIB=18944 MEMORYMAX_MIB=4096 CACHE_MIB=6144' ]"
check "q41 B D5"  "[ \"\$(footprint --weights q41 --arm B --mode D5)\" = 'FOOTPRINT_MIB=57344 MEMORYMAX_MIB=4096 CACHE_MIB=43520' ]"
check "iq3 A D1"  "[ \"\$(footprint --weights iq3 --arm A --mode D1)\" = 'FOOTPRINT_MIB=18954 MEMORYMAX_MIB=9984' ]"
check "iq3 R D5"  "[ \"\$(footprint --weights iq3 --arm R --mode D5)\" = 'FOOTPRINT_MIB=57344 MEMORYMAX_MIB=8192' ]"
check "q41exact A D5" "[ \"\$(footprint --weights q41exact --arm A --mode D5)\" = 'FOOTPRINT_MIB=57344 MEMORYMAX_MIB=47616' ]"
check "LB_CUDA_MIB=9000 recompute" "[ \"\$(LB_CUDA_MIB=9000 bash $L --print-footprint --weights q41 --arm A --mode D5)\" = 'FOOTPRINT_MIB=57344 MEMORYMAX_MIB=47320' ]"
check "MemFree large keeps the table" "[ \"\$(footprint --weights q41 --arm A --mode D5 --memfree-mib 90000)\" = 'FOOTPRINT_MIB=57344 MEMORYMAX_MIB=47616 MEMFREE_MIB=90000 TABLE_MIB=47616' ]"
check "MemFree 70250 clamps A D5 to 47210" "[ \"\$(footprint --weights q41 --arm A --mode D5 --memfree-mib 70250)\" = 'FOOTPRINT_MIB=57344 MEMORYMAX_MIB=47210 MEMFREE_MIB=70250 TABLE_MIB=47616' ]"
check "MemFree clamp reaches the dry-run" "[[ \"\$(bash $L --dry-run --weights q41 --arm A --mode D5 --memfree-mib 70250 2>&1)\" == *MemoryMax=47210M*MemoryHigh=46698M* ]]"
bash $L --print-footprint --weights q41 --arm A --mode D5 --memfree-mib 20000 >/dev/null 2>&1; check "MemFree too low exits 4" "[ \$? = 4 ]"
bash $L --print-footprint --weights iq3 --arm R --mode D1 >/dev/null 2>&1; check "iq3 R D1 exits 2" "[ $? = 2 ]"
bash $L --print-footprint --weights q41 --arm R --mode D5 >/dev/null 2>&1; check "q41 R D5 exits 2" "[ $? = 2 ]"

B="--weights q41 --arm B --mode D5"
check "arm B MemFree 72693 full cache" "[ \"\$(footprint \$B --memfree-mib 72693)\" = 'FOOTPRINT_MIB=57344 MEMORYMAX_MIB=4096 CACHE_MIB=43520 MEMFREE_MIB=72693 TABLE_MIB=4096 CACHE_TABLE_MIB=43520 CACHE_FULL=true' ]"
check "arm B MemFree 70465 full cache" "[ \"\$(footprint \$B --memfree-mib 70465)\" = 'FOOTPRINT_MIB=57344 MEMORYMAX_MIB=4096 CACHE_MIB=43520 MEMFREE_MIB=70465 TABLE_MIB=4096 CACHE_TABLE_MIB=43520 CACHE_FULL=true' ]"
check "arm B MemFree 70464 cache 43519" "[ \"\$(footprint \$B --memfree-mib 70464)\" = 'FOOTPRINT_MIB=57344 MEMORYMAX_MIB=4096 CACHE_MIB=43519 MEMFREE_MIB=70464 TABLE_MIB=4096 CACHE_TABLE_MIB=43520 CACHE_FULL=false' ]"
check "arm B MemFree 66985 cache 40040" "[ \"\$(footprint \$B --memfree-mib 66985)\" = 'FOOTPRINT_MIB=57344 MEMORYMAX_MIB=4096 CACHE_MIB=40040 MEMFREE_MIB=66985 TABLE_MIB=4096 CACHE_TABLE_MIB=43520 CACHE_FULL=false' ]"
check "arm B MemFree 66985 dry-run has --moe-cache-mib 40040" "[[ \"\$(bash $L --dry-run \$B --memfree-mib 66985 2>&1)\" == *'--moe-cache-mib 40040'* ]]"
bash $L --print-footprint $B --memfree-mib 28000 >/dev/null 2>&1; check "arm B MemFree 28000 exits 4" "[ \$? = 4 ]"
check "arm A MemFree 66985 prints no CACHE token" "[[ \"\$(footprint --weights q41 --arm A --mode D5 --memfree-mib 66985)\" != *CACHE* ]]"

mand=(-lm\ mmap -lzm\ on --no-repack "-ngl 99" "-fit off" "-c 32768" "-np 1" "-b 2048" "-fa on" "-t 16" "--cache-ram 0" "-ctxcp 4" "--host 127.0.0.1" "--port 8091" --metrics "-lv 4" "-cmoe" "MemoryMax=47616M" "MemorySwapMax=0" "MemoryHigh=47104M" "--unit=slotstream-spark-lb")
DRY=$(bash $L --dry-run --weights q41 --arm A --mode D5 2>&1)
for f in "${mand[@]}"; do
  check "dry-run has '$f'" "[[ \"\$DRY\" == *\"\$f\"* ]]"
done
check "dry-run has no --sleep-idle-seconds" "[[ \"\$DRY\" != *--sleep-idle-seconds* ]]"
check "dry-run has no --moe-cache-mib for arm A" "[[ \"\$DRY\" != *--moe-cache-mib* ]]"
DB=$(bash $L --dry-run --weights q41 --arm B --mode D5 2>&1)
check "arm B dry-run has --moe-cache-mib 43520" "[[ \"\$DB\" == *'--moe-cache-mib 43520'* ]]"
DR=$(bash $L --dry-run --weights iq3 --arm R --mode D5 2>&1)
check "arm R: -ub 512 and no -cmoe" "[[ \"\$DR\" == *'-ub 512'* && \"\$DR\" != *-cmoe* ]]"
check "script text avoids --sleep-idle-seconds" "! grep -q -e '--sleep-idle-seconds' $L"

check "q41 A dry-run has --no-op-offload" "[[ \"\$DRY\" == *--no-op-offload* ]]"
check "q41 B dry-run has --no-op-offload" "[[ \"\$DB\" == *--no-op-offload* ]]"
check "iq3 R dry-run lacks --no-op-offload" "[[ \"\$DR\" != *--no-op-offload* ]]"
check "print-flags q41 A D5" "[ \"\$(bash $L --print-flags --weights q41 --arm A --mode D5)\" = 'OP_OFFLOAD=false MEMFREE_STOP_MIB=14336' ]"
check "print-flags iq3 R D5" "[ \"\$(bash $L --print-flags --weights iq3 --arm R --mode D5)\" = 'OP_OFFLOAD=true MEMFREE_STOP_MIB=14336' ]"
check "print-flags --memfree-stop 0" "[[ \"\$(bash $L --print-flags --weights q41 --arm A --mode D5 --memfree-stop 0)\" == *MEMFREE_STOP_MIB=0 ]]"

W=deploy/lb/memfree_watch.sh
T=$(mktemp -d)
printf 'MemFree:         1048576 kB\nCached:  2097152 kB\n' > "$T/low"
printf 'MemFree:        83886080 kB\nCached:  2097152 kB\n' > "$T/high"
sleep 30 & P=$!
t0=$SECONDS; bash $W $P 14336 "$T/low.csv" "$T/low"; rc=$?; dt=$((SECONDS - t0))
check "watcher low MemFree exits 5 within 5 s" "[ $rc = 5 ] && [ $dt -le 5 ]"
check "watcher wrote LOG.stop" "[ -f $T/low.csv.stop ]"
check "watcher ended the pid" "! kill -0 $P 2>/dev/null"
sleep 2 & P=$!
bash $W $P 14336 "$T/high.csv" "$T/high"; rc=$?
check "watcher high MemFree exits 0" "[ $rc = 0 ] && [ ! -f $T/high.csv.stop ]"
check "watcher log has header and a row" "[ \$(wc -l < $T/high.csv) -ge 2 ] && head -1 $T/high.csv | grep -q '^epoch,memfree_mib'"
rm -rf "$T"

echo "llamacpp_t0: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
