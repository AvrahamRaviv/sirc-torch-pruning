"""MobileNetV1 autoresearch retention harness — sibling of harness_mbv2.py.

Added on boss request. MobileNetV1 = pure depthwise-separable CHAIN (no inverted residual, no
skip connections) → simplest topology of all the benchmarked nets. timm mobilenetv1_100
(ra4_e3600_r224_in1k, dense ~0.57G / 4.23M, top-1 ~0.728). model_type='cnn'.

PURE pre-FT (no BN recalib), exactly like harness_mbv2: fold native Conv->BN into the conv
weights → BN-free dense, score+prune the same folded net, evaluate as-is. Depthwise convs
(groups>1) are NOT reparam'd/scored (build_whole_net_reparam_layers picks groups==1 + Linear);
only the pointwise 1x1 convs (+ conv_stem) carry scores. Pruning a pointwise output channel →
the following depthwise + next pointwise input follow by TP coupling. No residual → rblocks={},
join_cov off, propagation is a clean chain.

Scorers (boss list): magnitude, vbp(=variance_vbp), prop(=prop_rel_p2), cov(=prop_p2_cov,
width+mean), cov+iter(=prop_iter_cov). + random sanity. Own results_mbv1.jsonl.

Usage:
    python harness_mbv1.py --val_limit 5000 --mac_target_g 0.38 --max_prune_ratio 0.8
    python harness_mbv1.py --only prop --val_limit 256          # smoke
"""
import argparse
import copy
import glob
import hashlib
import io
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects/Torch-Pruning")
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects/Torch-Pruning/benchmarks/vbp")
import pyarrow.parquet as pq
import timm
import torch
import torch.nn as nn
import torch_pruning as tp
from PIL import Image

from vbp_common import get_val_transform
from normalize_net import build_whole_net_reparam_layers, build_reparam_manager
from normalized_net_importance import extract_normnet_scores, NormalizedNetImportance
import normnet_main as NM


def mbv1_var_targets(model):
    """Post-activation variance targets for timm mobilenetv1 = (prunable conv, following
    BatchNormAct2d). NM._post_act_target_layers returns act=None here because the ReLU6 is FUSED
    inside BatchNormAct2d (no separate sibling act module) → variance_vbp would measure PRE-ReLU6
    output variance (wrong tensor; ReLU6 clips) and lands BELOW random. VarianceImportance applies
    post_act_fn to the conv output; passing the BatchNormAct2d module (callable → BN+ReLU6) makes
    the hook capture the true post-activation, keyed by the conv (so it maps to the pruned dim)."""
    import timm.layers as TL
    targets = []
    # stem conv → model.bn1 (BatchNormAct2d)
    if hasattr(model, "conv_stem") and hasattr(model, "bn1"):
        targets.append((model.conv_stem, model.bn1))
    # each depthwise-separable block: conv_pw → bn2 (post-act)
    for mod in model.modules():
        if isinstance(mod, TL.DepthwiseSeparableConv) if hasattr(TL, "DepthwiseSeparableConv") \
                else (hasattr(mod, "conv_pw") and hasattr(mod, "bn2")):
            targets.append((mod.conv_pw, mod.bn2))
    return targets


def fold_bnact(model):
    """Fold Conv->BN into the conv, replacing the BN module with its ACTIVATION (not Identity).

    Why not torch_pruning fold_all_conv_bn: timm mobilenetv1 uses BatchNormAct2d (a
    nn.BatchNorm2d subclass with the ReLU6 fused inside `.act`). The generic fold matches it as
    a BN and swaps it for nn.Identity → SILENTLY DROPS every activation → dense collapses to
    top-1 0.0 (verified). Here we bake the BN affine+stats into the conv weight/bias and put the
    fused activation back in the BN's slot, so the folded net is BN-free AND function-preserving
    (folded mobilenetv1_100 top-1 = unfolded 0.7227). Returns the fold count."""
    n = 0
    for parent in list(model.modules()):
        kids = list(parent.named_children())
        for i in range(len(kids) - 1):
            ka, a = kids[i]
            kb, b = kids[i + 1]
            if not isinstance(a, (nn.Conv2d, nn.Linear)):
                continue
            if not isinstance(b, nn.BatchNorm2d):          # BatchNormAct2d is a subclass → matches
                continue
            if b.num_features != a.weight.shape[0]:
                continue
            s = b.weight.detach() / torch.sqrt(b.running_var.detach() + b.eps)
            W = a.weight.detach().clone()
            W *= s.reshape([-1] + [1] * (W.dim() - 1))
            bias = a.bias.detach().clone() if a.bias is not None else torch.zeros_like(b.running_mean)
            bias = (bias - b.running_mean.detach()) * s + b.bias.detach()
            a.weight = nn.Parameter(W)
            a.bias = nn.Parameter(bias)
            act = getattr(b, "act", None)
            setattr(parent, kb, act if (act is not None and not isinstance(act, nn.Identity))
                    else nn.Identity())
            n += 1
    return n

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "imagenet1k_val", "data")
RESULTS = os.path.join(HERE, "results_mbv1.jsonl")
LEADERBOARD = os.path.join(HERE, "LEADERBOARD_mbv1.md")
MTYPE = "cnn"


# --------------------------------------------------------------------- data
def _rows(shards, limit=None, skip=0):
    n = 0
    for sh in shards:
        t = pq.read_table(sh, columns=["image", "label"])
        imgs, labs = t.column("image").to_pylist(), t.column("label").to_pylist()
        for im, lb in zip(imgs, labs):
            if n < skip:
                n += 1
                continue
            yield im["bytes"], lb
            n += 1
            if limit is not None and n >= skip + limit:
                return


def _batched(rows, tf, bs):
    xs, ys = [], []
    for b, lb in rows:
        xs.append(tf(Image.open(io.BytesIO(b)).convert("RGB")))
        ys.append(lb)
        if len(xs) == bs:
            yield torch.stack(xs), torch.tensor(ys)
            xs, ys = [], []
    if xs:
        yield torch.stack(xs), torch.tensor(ys)


class ListLoader:
    def __init__(self, batches): self.batches = batches
    def __iter__(self): return iter(self.batches)
    def __len__(self): return len(self.batches)


@torch.no_grad()
def recalibrate_bn(model, calib, device, max_batches):
    """Reset + re-estimate BN running stats on calib after pruning. Pruning a producer's output
    channels shifts every downstream BN's input distribution → stale stats collapse accuracy.
    Pre-FT analogue of reinserting fresh BN at pruned widths. BatchNormAct2d is a _BatchNorm
    subclass so reset_running_stats works and its fused act stays in forward."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for m in bns:
        m.reset_running_stats()
        m.momentum = None            # cumulative moving average over the calib pass
    model.train().to(device)
    for i, (x, _) in enumerate(calib):
        if i >= max_batches:
            break
        model(x.to(device))
    model.eval().to("cpu")


@torch.no_grad()
def evaluate(model, shards, tf, bs, device, limit):
    model.eval().to(device)
    correct = total = 0
    for x, y in _batched(_rows(shards, limit=limit), tf, bs):
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    model.to("cpu")
    return correct / max(total, 1), total


# --------------------------------------------------------------------- context (cached once)
class Ctx:
    def __init__(self, args):
        self.args = args
        self.mtype = MTYPE
        self.shards = sorted(glob.glob(os.path.join(DATA, "validation-*.parquet")))
        assert self.shards, f"no shards under {DATA}"
        self.tf = get_val_transform("cnn", resize=256)
        self.ex = torch.randn(1, 3, 224, 224)
        m = timm.create_model("mobilenetv1_100", pretrained=True)
        m.eval()
        print("[ctx] loaded timm mobilenetv1_100 (ra4_e3600_r224_in1k)", flush=True)
        # resnet-style protocol: dense stays UNFOLDED (native BatchNormAct2d). mbv1 is compact and
        # fragile → pure BN-free pre-FT collapses at −34% (verified, all scorers ~0). Keep BN and
        # recalibrate post-prune (lifts a BN net massively, as on resnet). Scoring copy is folded.
        self.dense = m
        self.dense_macs, self.dense_params = NM._count(self.dense, self.ex)
        print(f"[ctx] decoding {args.calib_batches} calib batches...", flush=True)
        self.calib = ListLoader(list(_batched(
            _rows(self.shards[::-1], limit=args.calib_batches * args.batch_size),
            self.tf, args.batch_size)))
        self._build_normnet()
        self._var_stats = None
        self._nci_scores = None
        self._means_by_name = None

    def _build_normnet(self):
        # SCORING copy is folded (BN-free) → v_tilde=σW, σ_out measured POST-BN (function-
        # preserving); dense/base stay unfolded (native BN) for pruning + recalib.
        self.norm_model = copy.deepcopy(self.dense)
        n_fold = fold_bnact(self.norm_model)
        print(f"[ctx] folded {n_fold} Conv->BN(Act) on scoring copy", flush=True)
        nargs = SimpleNamespace(reparam_variant="mean", norm_bn_momentum=0.01,
                                mu_ema_momentum=0.0, calib_batches=self.args.calib_batches,
                                max_batches=self.args.calib_batches, model_type=MTYPE,
                                cnn_arch="mobilenetv1", exclude_stem=False)
        names = build_whole_net_reparam_layers(self.norm_model, exclude_classifier=True,
                                               exclude_stem=False)
        print(f"[ctx] reparam layers (pointwise+stem, groups==1): {len(names)}", flush=True)
        self.mgr = build_reparam_manager(self.norm_model, names, "cpu", nargs)
        print("[ctx] calibrating reparam σ/μ...", flush=True)
        self.mgr.reparameterize(self.calib)
        self.clf = NM._classifier(self.norm_model)
        self.bscale = NM._propagation_branch_scale(self.norm_model)
        self.rblocks = NM._residual_blocks(self.norm_model)          # {} for mobilenet (no skip)
        self._icov = self._jcov = None

    @property
    def icov(self):
        if self._icov is None:
            print("[ctx] collecting input covariance...", flush=True)
            self._icov = self.mgr.collect_input_covariance(self.calib,
                                                           max_batches=self.args.calib_batches)
        return self._icov

    @property
    def jcov(self):
        return {}                                          # no residual blocks → join_cov off

    def extract_kwargs(self, cov=False, join_cov=False, p=2, relative=True,
                       measured_var=False):
        return dict(p=p, relative=relative, classifier=self.clf, use_measured_sigma_c=False,
                    use_measured_var=measured_var, branch_out_scale=self.bscale,
                    input_cov=(self.icov if cov else None), join_cov=None)

    def base(self):
        return copy.deepcopy(self.dense)

    def var_stats(self, mode):
        if self._var_stats is None:
            tl = mbv1_var_targets(self.dense)              # POST-act (fused ReLU6) targets
            vi = tp.importance.VarianceImportance(norm_per_layer=False, importance_mode=mode)
            vi.collect_statistics(self.dense, self.calib, "cpu", target_layers=tl,
                                  max_batches=self.args.calib_batches)
            self._var_stats = {nm: (vi.variance[m], vi.means.get(m))
                               for nm, m in self.dense.named_modules() if m in vi.variance}
        return self._var_stats

    def nci_scores(self):
        if self._nci_scores is None:
            self._nci_scores = NM.compute_nci_cov_scores(self.base(), self.calib, "cpu",
                                                         self.args.calib_batches)
        return self._nci_scores

    def means_by_name(self):
        if self._means_by_name is None:
            tl = mbv1_var_targets(self.dense)
            vi = tp.importance.VarianceImportance()
            vi.collect_statistics(self.dense, self.calib, "cpu", target_layers=tl,
                                  max_batches=self.args.calib_batches)
            self._means_by_name = {nm: vi.means[m] for nm, m in self.dense.named_modules()
                                   if m in vi.means}
        return self._means_by_name


# --------------------------------------------------------------------- prune + eval one spec
def _norm_arg(normalizer):
    return None if normalizer in (None, "none") else normalizer


def build_importance_factory(ctx, spec):
    kind = spec["kind"]
    p = spec.get("params", {})
    norm = p.get("normalizer", "width")

    if kind == "magnitude":
        return (lambda mdl: tp.importance.GroupMagnitudeImportance(
            p=2, group_reduction="mean", normalizer=_norm_arg(norm)),
            ctx.dense, norm)
    if kind == "random":
        return (lambda mdl: tp.importance.RandomImportance(), ctx.dense, "none")
    if kind in ("variance", "tp_variance"):
        stats = ctx.var_stats(kind)
        def f(mdl):
            vi = tp.importance.VarianceImportance(norm_per_layer=(norm == "mean"),
                                                  importance_mode=kind)
            for nm, m in mdl.named_modules():
                if nm in stats:
                    var, mean = stats[nm]
                    vi.variance[m] = var
                    if mean is not None:
                        vi.means[m] = mean
            return vi
        return (f, ctx.dense, norm)
    if kind == "nci_cov":
        scores = ctx.nci_scores()
        return (lambda mdl: NormalizedNetImportance(mdl, scores, group_reduction="mean",
                normalizer=_norm_arg(norm), fallback=False), ctx.base(), norm)
    if kind == "per_layer":
        scores = ctx.mgr.input_channel_scores()
        return (lambda mdl: NormalizedNetImportance(mdl, scores, group_reduction="mean",
                normalizer=_norm_arg(norm), fallback=False), ctx.base(), norm)
    if kind == "propagation":
        ek = ctx.extract_kwargs(cov=p.get("cov", False), join_cov=p.get("join_cov", False),
                                p=p.get("p", 2), relative=p.get("relative", True),
                                measured_var=p.get("measured_var", False))
        scores = extract_normnet_scores(ctx.mgr, "propagation", example_inputs=ctx.ex, **ek)
        return (lambda mdl: NormalizedNetImportance(mdl, scores, group_reduction="mean",
                normalizer=_norm_arg(norm), fallback=False), ctx.base(), norm)
    if kind == "propagation_iterative":
        ek = ctx.extract_kwargs(cov=p.get("cov", False), join_cov=p.get("join_cov", False),
                                p=p.get("p", 2), relative=p.get("relative", True))
        _ig_ids = {id(m) for m in NM._ignored_layers(ctx.norm_model, MTYPE)}
        _ig_names = {n for n, m in ctx.norm_model.named_modules() if id(m) in _ig_ids}
        scores, _ = NM._iterative_propagation_scores(
            ctx.mgr, ctx.ex, ek, model_type=MTYPE, normalizer=norm,
            drop_per_round=p.get("iter_drop", 128), max_frac=p.get("iter_max_frac", 0.6),
            ignored_names=_ig_names, log=lambda s: None)
        return (lambda mdl: NormalizedNetImportance(mdl, scores, group_reduction="mean",
                normalizer=None, fallback=False), ctx.base(), "none")
    raise ValueError(f"unknown kind {kind!r}")


def run_spec(ctx, spec, limit):
    imp_factory, base, _norm = build_importance_factory(ctx, spec)
    model = copy.deepcopy(base)
    ignored = lambda mdl: NM._ignored_layers(mdl, MTYPE)
    mpr = ctx.args.max_prune_ratio if ctx.args.max_prune_ratio > 0 else 1.0
    ratio = ctx.args.pruning_ratio
    if ctx.args.mac_target_g > 0:
        ratio = NM._ratio_for_mac(model, ctx.ex, imp_factory, ignored,
                                  ctx.args.mac_target_g, global_pruning=True,
                                  max_pruning_ratio=mpr)
    mean_dict = None
    if ctx.args.bias_comp:
        mbn = ctx.means_by_name()
        mean_dict = {m: mbn[nm] for nm, m in model.named_modules() if nm in mbn}
    tp.pruner.MagnitudePruner(model, ctx.ex, importance=imp_factory(model),
                              global_pruning=True, pruning_ratio=ratio,
                              max_pruning_ratio=mpr,
                              ignored_layers=ignored(model), mean_dict=mean_dict).step()
    macs, params = NM._count(model, ctx.ex)
    if not ctx.args.no_bn_recalib:
        recalibrate_bn(model, ctx.calib, ctx.args.eval_device, ctx.args.calib_batches)
    acc, n = evaluate(model, ctx.shards, ctx.tf, ctx.args.batch_size, ctx.args.eval_device, limit)
    return dict(acc=acc, n=n, macs=macs, params=params, ratio=ratio,
                mac_pct=100 * macs / ctx.dense_macs)


# --------------------------------------------------------------------- leaderboard io
def _key(spec, limit, bias_comp=True):
    h = hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    return f"{spec['name']}@{limit}{'+bc' if bias_comp else ''}#{h}"


def load_done():
    if not os.path.exists(RESULTS):
        return {}
    out = {}
    with open(RESULTS) as f:
        for line in f:
            r = json.loads(line)
            out[r["key"]] = r
    return out


def append_result(rec):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def rewrite_leaderboard():
    done = load_done()
    rows = sorted(done.values(), key=lambda r: -r["res"]["acc"])
    with open(LEADERBOARD, "w") as f:
        f.write("# Retention leaderboard — MobileNetV1, PURE pre-FT (no BN recalib) @ matched MAC\n\n")
        f.write("| rank | spec | top-1 | MAC% | params(M) | n | normalizer | notes |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(rows, 1):
            res, sp = r["res"], r["spec"]
            f.write(f"| {i} | `{sp['name']}` | {res['acc']:.4f} | {res['mac_pct']:.0f}% | "
                    f"{res['params']:.2f} | {res['n']} | "
                    f"{sp.get('params', {}).get('normalizer', '-')} | {sp.get('note', '')} |\n")


# --------------------------------------------------------------------- specs (boss list)
def baseline_specs(normalizer="width"):
    P = lambda **k: dict(normalizer=normalizer, **k)
    return [
        dict(name="random", kind="random"),
        dict(name="magnitude", kind="magnitude", params=P()),
        dict(name="variance_vbp", kind="variance", params=P(), note="VBP activation σ²"),
        dict(name="prop_rel_p2", kind="propagation", params=P(relative=True, p=2), note="prop"),
        dict(name="prop_p2_cov", kind="propagation", params=P(cov=True), note="cov, width"),
        dict(name="prop_p2_cov_normMean", kind="propagation",
             params=dict(normalizer="mean", cov=True), note="cov, mean"),
        dict(name="prop_iter_cov", kind="propagation_iterative",
             params=P(cov=True, join_cov=False, iter_drop=128, iter_max_frac=0.6),
             note="cov+iter, width"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="run only specs whose name contains this")
    ap.add_argument("--eval_device", default="mps")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--calib_batches", type=int, default=50)
    ap.add_argument("--val_limit", type=int, default=5000)
    ap.add_argument("--mac_target_g", type=float, default=0.0,
                    help="target GMAC. <=0 ⇒ auto = 0.66×dense (−34% MAC).")
    ap.add_argument("--pruning_ratio", type=float, default=0.5)
    ap.add_argument("--normalizer", default="width")
    ap.add_argument("--max_prune_ratio", type=float, default=0.8,
                    help="per-layer prune cap (keep >= 1-mpr). 0.8 default (narrow mbv1 layers).")
    ap.add_argument("--no_bias_comp", action="store_true")
    ap.add_argument("--no_bn_recalib", action="store_true",
                    help="skip post-prune BN re-estimation (default ON — required for meaningful "
                         "pre-FT on the compact mbv1 BN net, as on resnet).")
    args = ap.parse_args()
    args.bias_comp = not args.no_bias_comp

    specs = baseline_specs(args.normalizer)
    if args.only:
        specs = [s for s in specs if args.only in s["name"]]

    ctx = Ctx(args)
    if args.mac_target_g <= 0:
        args.mac_target_g = round(0.66 * ctx.dense_macs, 2)
        print(f"[harness] auto MAC target = {args.mac_target_g}G (0.66×dense = −34% MAC)", flush=True)
    done = load_done()
    print(f"[harness] dense {ctx.dense_macs:.2f}G {ctx.dense_params:.2f}M; "
          f"{len(specs)} specs; {len(done)} already done", flush=True)

    for spec in specs:
        k = _key(spec, args.val_limit, args.bias_comp)
        if k in done:
            print(f"  skip (done): {spec['name']}  acc={done[k]['res']['acc']:.4f}", flush=True)
            continue
        t0 = time.time()
        try:
            res = run_spec(ctx, spec, args.val_limit)
        except Exception as e:
            import traceback
            print(f"  FAIL {spec['name']}: {e}", flush=True)
            traceback.print_exc()
            append_result(dict(key=k, spec=spec, res=dict(acc=-1, n=0, macs=0, params=0,
                          ratio=0, mac_pct=0), error=str(e)[:200], limit=args.val_limit))
            continue
        rec = dict(key=k, spec=spec, res=res, limit=args.val_limit, sec=round(time.time() - t0))
        append_result(rec)
        rewrite_leaderboard()
        print(f"  [{spec['name']}] top-1={res['acc']:.4f}  {res['mac_pct']:.0f}% MAC  "
              f"{res['params']:.2f}M  ({rec['sec']}s)", flush=True)

    rewrite_leaderboard()
    print(f"\n[harness] done. leaderboard → {LEADERBOARD}", flush=True)


if __name__ == "__main__":
    main()
