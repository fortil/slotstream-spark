import argparse
import sys

from . import ALLOWED, decided, get, load, validate


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slotstream_spark.config")
    ap.add_argument("--get")
    ap.add_argument("--require")
    ap.add_argument("--require-all", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--path")
    a = ap.parse_args(argv)
    cfg = load(a.path)
    if a.validate:
        errs = validate(cfg)
        if errs:
            print("\n".join(errs))
            return 1
    if a.get:
        print(get(a.get, cfg))
    keys = []
    if a.require:
        keys = [k.strip() for k in a.require.split(",") if k.strip()]
    elif a.require_all:
        keys = list(ALLOWED)
    if keys:
        bad = [k for k in keys if not decided(k, cfg)]
        if bad:
            print("undecided: " + ",".join(bad))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
