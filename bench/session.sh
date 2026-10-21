#!/bin/bash
# Spark: the F3 session job. Start it with
#   tools/spark/jobs.sh start f3-session --heavy f3-session auto 14400 --memcg slotstream-spark-lb -- bash bench/session.sh
# (the guard refuses it after ~19:45 and between 00:00 and 04:00).
exec bash tools/spark/snap.sh f3-session tools/spark/py.sh bench/session.py
