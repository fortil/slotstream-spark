"""Counters for /metrics, in the Prometheus text format. Thread-safe; nothing here touches the engine."""
import threading
from collections import defaultdict


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests = defaultdict(int)  # (route, status) -> n
        self.counters = defaultdict(float)
        self.gauges = {}

    def request(self, route, status):
        with self._lock:
            self.requests[(route, int(status))] += 1

    def inc(self, name, value=1):
        with self._lock:
            self.counters[name] += value

    def gauge(self, name, value):
        with self._lock:
            self.gauges[name] = value

    def get(self, name):
        with self._lock:
            return self.counters.get(name, 0)

    def generation(self, result, queue_wait_s):
        """TTFT split into queue wait and prefill; decode tok/s is the last generation's."""
        with self._lock:
            self.counters["generations_total"] += 1
            self.counters["queue_wait_seconds_sum"] += queue_wait_s
            self.counters["prefill_seconds_sum"] += result.prefill_seconds
            self.counters["prompt_tokens_total"] += result.prompt_tokens
            self.counters["decode_tokens_total"] += result.decode_tokens
            self.counters["reused_tokens_total"] += result.reused_prefix_tokens
            if result.decode_seconds > 0 and result.decode_tokens > 0:
                self.gauges["decode_tokens_per_second"] = result.decode_tokens / result.decode_seconds

    def render(self, snapshot, queued=0, busy=False):
        with self._lock:
            lines = []
            for (route, status), n in sorted(self.requests.items()):
                lines.append(f'slotstream_requests_total{{route="{route}",status="{status}"}} {n}')
            for name, v in sorted(self.counters.items()):
                lines.append(f"slotstream_{name} {v:g}")
            for name, v in sorted(self.gauges.items()):
                lines.append(f"slotstream_{name} {v:g}")
        lines.append(f"slotstream_pool_hits_total {snapshot.pool_hits}")
        lines.append(f"slotstream_footprint_bytes {snapshot.footprint_bytes}")
        lines.append(f'slotstream_governor_mode{{mode="{snapshot.governor_mode}"}} 1')
        lines.append(f"slotstream_queue_depth {queued}")
        lines.append(f"slotstream_generation_busy {1 if busy else 0}")
        return "\n".join(lines) + "\n"
