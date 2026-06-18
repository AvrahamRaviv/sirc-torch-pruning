"""Cross-machine propagation-score probe (debug).

Same command on cluster + local → compare per-layer score mean/cv on an IDENTICAL
fixed input tensor, to isolate score divergence (weights / torch-version / calib-data).

Usage (identical both sides except --ckpt):
  python _probe_scores.py --ckpt <mobilenet_v2 weights .pth> --fixed <calib32.pt> [--full]

--fixed is the shared fixed input tensor (e.g. calib32.pt, 32x3x224x224, fp16 ok).
Both machines MUST load the same file so the only remaining variables are the
checkpoint weights and the torch/library numerics.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch
import torchvision.models as tv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root
sys.path.insert(0, HERE)

import normnet_main as NM
from normalize_net import build_whole_net_reparam_layers, build_reparam_manager
from normalized_net_importance import extract_normnet_scores

PROBE_LAYERS = ["features.0.0", "features.5.conv.0.0", "features.10.conv.0.0",
                "features.14.conv.0.0", "features.17.conv.0.0", "features.18.0"]


class _OneBatch:
    def __init__(self, x):
        self.b = [(x, torch.zeros(x.shape[0], dtype=torch.long))]

    def __iter__(self):
        return iter(self.b)

    def __len__(self):
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="mobilenet_v2 weights .pth")
    ap.add_argument("--fixed", required=True, help="shared fixed input tensor (calib32.pt)")
    ap.add_argument("--full", action="store_true", help="print every layer, not just probes")
    ap.add_argument("--debug", action="store_true",
                    help="dump per-layer intermediates (raw |v|, sigma_x, sigma_out_x, seed) "
                         "to localize the divergence stage: weights vs calibration vs propagation")
    ap.add_argument("--no_measured", action="store_true",
                    help="use_measured_var=False → exact column-stochastic cov denom "
                         "(mass-conserving). If this matches across machines but the default "
                         "doesn't, measured_var is the platform-fragile (non-stochastic) term.")
    ap.add_argument("--no_cov", action="store_true", help="drop input_cov numerator too")
    args = ap.parse_args()

    print("torch", torch.__version__, "| torchvision", tv.__name__ and __import__("torchvision").__version__)
    x = torch.load(args.fixed, map_location="cpu").float()
    print("input", tuple(x.shape), "sum %.6f" % float(x.sum()), "std %.6f" % float(x.std()))

    m = tv.mobilenet_v2(weights=None)
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
    r = m.load_state_dict(sd, strict=False)
    print("load: missing", len(r.missing_keys), "unexpected", len(r.unexpected_keys))
    m.eval()

    calib = _OneBatch(x)
    ex = x[:1]
    nargs = SimpleNamespace(reparam_variant="mean", norm_bn_momentum=0.01, mu_ema_momentum=0.0,
                            calib_batches=1, max_batches=1, model_type="cnn",
                            cnn_arch="mobilenet_v2", exclude_stem=False)
    names = build_whole_net_reparam_layers(m, exclude_classifier=True, exclude_stem=False)
    mgr = build_reparam_manager(m, names, "cpu", nargs)
    mgr.reparameterize(calib)

    if args.debug:
        # STAGE diagnostics. Compare these across machines top-down: the FIRST stage that
        # differs is the culprit.
        #   |v| (raw weight L2)  → weights only; MUST match if checkpoints identical.
        #   sigma_x / sigma_out_x → calibration+forward; platform/torch-numerics sensitive.
        #   seed (terminal)       → classifier_seed = f(W, sigma_out_x).
        from torch_pruning.utils.normnet_importance import _classifier_seed  # noqa
        print("=== STAGE 1-2: weights (|v|) + calibration (sigma) per layer ===")
        for name, rp in mgr._reparam_modules.items():
            v = rp.v.detach().float()
            sx = rp.sigma_x.detach().float()
            so = rp.sigma_out_x.detach().float()
            print(f"  {name:24s} |v|={float(v.flatten().norm()):.6e} "
                  f"sx.mean={float(sx.mean()):.6e} so.mean={float(so.mean()):.6e} "
                  f"so.cv={float(so.std()/(so.mean()+1e-12)):.4f}")

    icov = mgr.collect_input_covariance(calib, max_batches=1)
    if args.debug:
        print("=== STAGE 3: input covariance diag mean per layer ===")
        for name, S in icov.items():
            d = S.detach().float().diagonal()
            print(f"  {name:24s} cov.diag.mean={float(d.mean()):.6e} cov.absmax={float(S.abs().max()):.6e}")
        _topo = mgr.build_propagation_topology(ex, p=2)
        _seed = _classifier_seed(mgr, _topo, NM._classifier(m), 2)
        if _seed:
            for tn, sv in _seed.items():
                sv = sv.float()
                print(f"=== STAGE 4: seed[{tn}] mean={float(sv.mean()):.6e} "
                      f"cv={float(sv.std()/(sv.mean()+1e-12)):.4f} ===")
    ek = dict(p=2, relative=True, classifier=NM._classifier(m), use_measured_sigma_c=False,
              use_measured_var=(not args.no_measured), branch_out_scale=None,
              input_cov=(None if args.no_cov else icov), join_cov=None)
    print(f"config: use_measured_var={not args.no_measured} input_cov={not args.no_cov}")
    sc = extract_normnet_scores(mgr, "propagation", example_inputs=ex, **ek)

    layers = list(sc.keys()) if args.full else PROBE_LAYERS
    print("scores (mean | cv | min | max):")
    for n in layers:
        if n not in sc:
            continue
        s = sc[n].float()
        cv = float(s.std() / (s.mean() + 1e-12))
        print(f"  {n:24s} {float(s.mean()):.4e} | {cv:.4f} | "
              f"{float(s.min()):.4e} | {float(s.max()):.4e}")


if __name__ == "__main__":
    main()
