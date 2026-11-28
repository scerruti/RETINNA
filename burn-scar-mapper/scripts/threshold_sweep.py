"""Multi-fire threshold check.

    python scripts/threshold_sweep.py run CKPT FRAP:523 FRAP:1075 ...   # one inference pass per fire, cached
    python scripts/threshold_sweep.py report --tune FRAP:523,FRAP:1075 # pick threshold on tune fires, score the rest

`run` saves outputs/sweep/<fire>.npz (probabilities, perimeter mask, dNBR mask, validity) so thresholds can be
re-scored without re-downloading imagery. `report` also gives a leave-one-fire-out score for every fire.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from burnmap import analysis, imagery, inference, perimeters
from burnmap.model import load_checkpoint

OUT = Path(__file__).resolve().parents[1] / "outputs" / "sweep"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]


def run(ckpt: str, fire_ids: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(ckpt)
    for fid in fire_ids:
        path = OUT / f"{fid.replace(':', '_')}.npz"
        if path.exists():
            print(f"{fid}: cached")
            continue
        t0 = time.time()
        try:
            fire = perimeters.get(fid)
            pre, post = imagery.fire_scenes(fire.geometry, fire.start, fire.end)
            x = inference.normalize(post.nir, post.swir, model.channel_means, model.channel_stds)
            prob = inference.predict_probabilities(model.net, x, device=model.device)
            ref = analysis.rasterize(imagery.to_crs(fire.geometry, post.crs), prob.shape, post.transform)
            dnbr, _ = analysis.dnbr_mask(pre.nir, pre.swir, post.nir, post.swir)
            np.savez_compressed(path, prob=prob.astype(np.float16), ref=ref, valid=post.valid,
                                both=post.valid & pre.valid, dnbr=dnbr & post.valid & pre.valid,
                                pre_nbr=analysis.nbr(pre.nir, pre.swir).astype(np.float16))
            (OUT / f"{fid.replace(':', '_')}.json").write_text(json.dumps({
                "fire": fid, "name": fire.name, "year": fire.year, "acres": fire.acres,
                "pre": str(pre.date), "post": str(post.date),
                "pre_clear": pre.valid_fraction, "post_clear": post.valid_fraction}))
            print(f"{fid} {fire.name} {fire.year}: done in {time.time() - t0:.0f}s "
                  f"(pre {pre.date}, post {post.date}, post clear {post.valid_fraction:.0%})", flush=True)
        except Exception as e:  # keep going; report which fires failed
            print(f"{fid}: FAILED {type(e).__name__}: {e}", flush=True)


def _load(fid):
    stem = fid.replace(":", "_")
    d = np.load(OUT / f"{stem}.npz")
    return json.loads((OUT / f"{stem}.json").read_text()), {k: d[k] for k in d.files}


def _score(d, t):
    pred = (d["prob"].astype(np.float32) > t) & d["valid"]
    return analysis.compare(pred, d["ref"], d["valid"]), analysis.acres(pred)


def report(tune: list[str]) -> None:
    fires = sorted(p.stem.replace("_", ":", 1) for p in OUT.glob("*.npz"))
    data = {f: _load(f) for f in fires}
    iou = {f: {t: _score(d, t)[0].iou for t in THRESHOLDS} for f, (_, d) in data.items()}

    print("IoU vs perimeter by threshold")
    print(f"{'fire':<22}" + "".join(f"{t:>6}" for t in THRESHOLDS) + "  best   dNBR")
    for f, (m, d) in data.items():
        dn = analysis.compare(d["dnbr"], d["ref"], d["both"]).iou
        best = max(iou[f], key=iou[f].get)
        print(f"{m['name'][:12] + ' ' + str(m['year']):<22}" + "".join(f"{iou[f][t]:6.2f}" for t in THRESHOLDS)
              + f"  {best:<5}  {dn:.2f}")

    def pick(fs):
        return max(THRESHOLDS, key=lambda t: np.mean([iou[f][t] for f in fs]))

    def row(f, t):
        m, d = data[f]
        a, ac = _score(d, t)
        dn = analysis.compare(d["dnbr"], d["ref"], d["both"])
        print(f"  {m['name'][:12] + ' ' + str(m['year']):<20} thr {t:<5} IoU {a.iou:.2f} P {a.precision:.2f} "
              f"R {a.recall:.2f} | dNBR IoU {dn.iou:.2f} P {dn.precision:.2f} R {dn.recall:.2f} "
              f"| model {ac:,.0f} ac, perimeter {analysis.acres(d['ref'] & d['valid']):,.0f} ac")

    if tune:
        t = pick(tune)
        print(f"\nThreshold picked on {', '.join(tune)}: {t}")
        print("Held-out fires:")
        for f in fires:
            if f not in tune:
                row(f, t)
    print("\nLeave-one-fire-out (threshold picked on all other fires):")
    for f in fires:
        row(f, pick([g for g in fires if g != f]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2], sys.argv[3:])
    elif cmd == "report":
        tune = sys.argv[sys.argv.index("--tune") + 1].split(",") if "--tune" in sys.argv else []
        report(tune)
