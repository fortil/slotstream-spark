import sys  # shim: the module lives in slotstream_spark.host.eligibility; alias it so patches reach the real one
from slotstream_spark.host import eligibility as _m; sys.modules[__name__] = _m  # noqa: E702
