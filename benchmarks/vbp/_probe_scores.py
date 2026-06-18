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
    icov = mgr.collect_input_covariance(calib, max_batches=1)
    ek = dict(p=2, relative=True, classifier=NM._classifier(m), use_measured_sigma_c=False,
              use_measured_var=True, branch_out_scale=None, input_cov=icov, join_cov=None)
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
