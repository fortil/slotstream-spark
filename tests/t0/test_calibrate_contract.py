import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import calibrate_contract as cc  # noqa: E402

TOML = ROOT / "slotstream_spark" / "config" / "numeric_contract.toml"
LIMITS = ROOT / "tests" / "gates" / "parity31_limits.json"
NL, H, K, V, T = 3, 8, 4, 16, 5


def test_rule_reproduces_the_parity31_limits():
    d = json.loads(LIMITS.read_text())
    assert len(d["layers"]) == 4
    for x in d["layers"]:
        lim, sep = cc.rule(x["spread"], x["control"])
        assert abs(lim - x["limit"]) < 1e-12 and sep == x["separable"]


def test_rule_edges():
    assert cc.rule(0.01, 0.039)[1] is False          # c < 4 s
    assert cc.rule(0.01, 0.045)[1] is False          # c >= 4 s but c < 2 L (L = 0.03)
    assert cc.rule(0.01, 0.4) == (0.03, True)


def write_dump(root, pid, layers, ids, logits, t=T):
    d = Path(root) / "dev" / pid
    d.mkdir(parents=True)
    (d / "tokens.json").write_text(json.dumps({"id": pid, "set": "dev", "ids": list(range(t))}))
    (d / "DONE").write_text("ok\n")
    for L in range(NL):
        layers[L].astype(np.float32).tofile(d / f"layer_{L:02d}_out.f32")
        ids[L].astype(np.int32).tofile(d / f"router_{L:02d}_ids.i32")
    logits.astype(np.float32).tofile(d / "logits.f32")


def synth(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for pid in ("dev0", "dev1"):
        g = [rng.standard_normal((T, H)) for _ in range(NL)]
        gid = [np.stack([rng.permutation(32)[:K] for _ in range(T)]) for _ in range(NL)]
        gl = rng.standard_normal((T, V))
        fp = [x + 0.01 * (k + 1) * rng.standard_normal(x.shape) for k, x in enumerate(g)]
        fid = [i.copy() for i in gid]
        fid[1][0, 0] = 100  # one id out of T*K differs
        fl = gl + 0.01 * rng.standard_normal(gl.shape)
        sw = [x + 0.5 * rng.standard_normal(x.shape) for x in g]
        sid = [np.stack([rng.permutation(32)[:K] for _ in range(T)]) + 1000 for _ in range(NL)]
        ne = [x + 0.7 * rng.standard_normal(x.shape) for x in g]
        write_dump(tmp_path / "g", pid, g, gid, gl)
        write_dump(tmp_path / "f", pid, fp, fid, fl)
        write_dump(tmp_path / "s", pid, sw, sid, gl)
        write_dump(tmp_path / "n", pid, ne, gid, gl)
        out[pid] = dict(g=g, fp=fp, sw=sw, ne=ne, gid=gid, fid=fid, sid=sid, gl=gl, fl=fl)
    return out


def test_measure_matches_direct_computation(tmp_path):
    S = synth(tmp_path)
    m = cc.measure(tmp_path / "g", tmp_path / "f", tmp_path / "s", tmp_path / "n",
                   nlayers=NL, hidden=H, topk=K, vocab=V)
    assert m["prompts"] == ["dev0", "dev1"]
    for L in range(NL):
        # rel_l2(a, b) normalises by b: the spread is rel_l2(bf16, fp32), as in ref_layerwise --parity31-calibrate
        sp = [np.linalg.norm(S[p]["g"][L].astype(np.float32).astype(float) - S[p]["fp"][L].astype(np.float32).astype(float))
              / np.linalg.norm(S[p]["fp"][L].astype(np.float32).astype(float)) for p in S]
        assert m["layers"][L]["spread_max"] == pytest.approx(max(sp), rel=1e-6)
        ctl = [np.linalg.norm(S[p][k][L].astype(np.float32).astype(float) - S[p]["g"][L].astype(np.float32).astype(float))
               / np.linalg.norm(S[p]["g"][L].astype(np.float32).astype(float)) for p in S for k in ("sw", "ne")]
        assert m["layers"][L]["control_min"] == pytest.approx(min(ctl), rel=1e-6)
    # routing: only layer 1 of the fp32 dump differs, by one id in T*K
    assert m["layers"][0]["routing_spread_max"] == 0.0
    assert m["layers"][1]["routing_spread_max"] == pytest.approx(1 / (T * K))
    assert m["layers"][2]["routing_control_min"] == pytest.approx(1.0)
    # logits: brute-force KL(ref || other)
    kls, agree = [], []
    for p in S:
        r, o = S[p]["gl"].astype(np.float32).astype(float), S[p]["fl"].astype(np.float32).astype(float)
        pr, po = np.exp(r) / np.exp(r).sum(1, keepdims=True), np.exp(o) / np.exp(o).sum(1, keepdims=True)
        kls += list((pr * np.log(pr / po)).sum(1))
        agree += list(r.argmax(1) == o.argmax(1))
    assert m["logits"]["kl_mean"] == pytest.approx(np.mean(kls), rel=1e-6)
    assert m["logits"]["kl_max"] == pytest.approx(np.max(kls), rel=1e-6)
    assert m["logits"]["top1"] == pytest.approx(np.mean(agree))
    assert m["logits"]["positions"] == 2 * T


def calib(n, good=True, top1=1.0, kl_mean=0.001, kl_p99=0.01, kl_max=0.05):
    layers = [{"layer": k, "spread_max": 0.01 * (k + 1), "control_min": 0.4 if good else 0.001,
               "routing_spread_max": 0.02, "routing_control_min": 0.5 if k else 0.0} for k in range(n)]
    return {"prompts": ["dev0"], "layers": layers,
            "logits": {"top1": top1, "kl_mean": kl_mean, "kl_p99": kl_p99, "kl_max": kl_max, "positions": 10}}


def run_apply(tmp_path, c):
    cj, toml, rec = tmp_path / "c.json", tmp_path / "nc.toml", tmp_path / "rec.md"
    cj.write_text(json.dumps(c))
    toml.write_text(TOML.read_text())
    rec.write_text("Decision: x\n")
    p = subprocess.run([sys.executable, str(TOOLS / "calibrate_contract.py"), "apply", str(cj), "--toml", str(toml),
                        "--record", str(rec)], capture_output=True, text=True)
    return p, toml, rec


def test_apply_fills_arrays_and_keeps_other_keys(tmp_path):
    p, toml, rec = run_apply(tmp_path, calib(5))
    assert p.returncode == 0, p.stderr
    new, old = tomllib.loads(toml.read_text()), tomllib.loads(TOML.read_text())
    assert cc._strip(new) == cc._strip(old)
    L = new["layer"]
    assert len(L["limits"]) == 5 and all(abs(l - (2 * s + 0.01)) < 1e-12 for l, s in zip(L["limits"], L["spread"]))
    assert L["separable"] == [True] * 5
    assert new["routing"]["separable"][0] is False and new["routing"]["separable"][1] is True
    assert "Calibration status: OK" in rec.read_text()


def test_apply_three_separable_layers_fails_and_keeps_the_toml(tmp_path):
    p, toml, rec = run_apply(tmp_path, calib(3))
    assert p.returncode == 1 and "DESIGN GATE FAILED" in p.stderr
    assert toml.read_text() == TOML.read_text()
    assert "FAILED" in rec.read_text()


def test_apply_oracle_top1_090_breaks_the_floor(tmp_path):
    p, toml, _ = run_apply(tmp_path, calib(5, top1=0.90))
    assert p.returncode == 1 and "top1_min" in p.stderr
    assert toml.read_text() == TOML.read_text()


def test_apply_oracle_kl_mean_002_breaks_the_ceiling(tmp_path):
    p, toml, _ = run_apply(tmp_path, calib(5, kl_mean=0.02))
    assert p.returncode == 1 and "kl_mean_max_nats" in p.stderr
    assert toml.read_text() == TOML.read_text()


def test_apply_writes_logits_limits_by_the_rule(tmp_path):
    p, toml, _ = run_apply(tmp_path, calib(5, top1=0.97, kl_mean=0.01, kl_p99=0.1, kl_max=0.3))
    assert p.returncode == 0, p.stderr
    new = tomllib.loads(toml.read_text())
    L = new["logits"]
    assert abs(L["top1_min"] - 0.94) < 1e-12
    for key, v in (("kl_mean_max_nats", 0.03), ("kl_p99_max_nats", 0.3), ("kl_max_nats", 0.9)):
        assert abs(L[key] - v) < 1e-12
    assert (L["o_top1"], L["o_kl_mean_nats"], L["o_kl_p99_nats"], L["o_kl_max_nats"]) == (0.97, 0.01, 0.1, 0.3)
    assert L["o_positions"] == 10 and isinstance(L["o_positions"], int)
    assert new["fixture"]["kl_max_nats"] == L["kl_max_nats"] == pytest.approx(0.9)
    assert cc._strip(new) == cc._strip(tomllib.loads(TOML.read_text()))


def test_apply_replaces_the_calibrated_section(tmp_path):
    p, toml, rec = run_apply(tmp_path, calib(5))
    assert p.returncode == 0, p.stderr
    toml.write_text(TOML.read_text())
    p = subprocess.run([sys.executable, str(TOOLS / "calibrate_contract.py"), "apply", str(tmp_path / "c.json"),
                        "--toml", str(toml), "--record", str(rec)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    text = rec.read_text()
    assert text.count("## Calibrated arrays") == 1 and text.count("Calibration status:") == 1


def test_rule_logits_on_the_measured_pair():
    cfg = tomllib.loads(TOML.read_text())["logits"]
    lim, bad = cc.rule_logits({"top1": 0.9448818897637795, "kl_mean": 0.012410261524308493,
                               "kl_p99": 0.14195519072051693, "kl_max": 0.3326383506586301}, cfg)
    assert bad == []
    for k, v in (("top1_min", 0.889763779527559), ("kl_mean_max_nats", 0.03723078457292548),
                 ("kl_p99_max_nats", 0.4258655721615508), ("kl_max_nats", 0.9979150519758904)):
        assert abs(lim[k] - v) < 1e-12, k


def test_apply_non_separable_layers_still_ok_if_enough(tmp_path):
    c = calib(6)
    c["layers"][5]["control_min"] = 0.001
    p, toml, _ = run_apply(tmp_path, c)
    assert p.returncode == 0
    assert tomllib.loads(toml.read_text())["layer"]["separable"] == [True] * 5 + [False]
