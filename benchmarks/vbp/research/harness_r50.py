"""ResNet-50 autoresearch retention harness — sibling of harness.py (convnext).

Same karpathy/autoresearch pattern: immutable eval harness + pre-FT retention top-1 at a
matched MAC target. Arch deltas vs harness.py: torchvision resnet50; native Conv->BN folded
BEFORE reparam (so σ_out is measured POST-BN, function-preserving); model_type='cnn';
branch_out_scale=None (no layer-scale gamma); own results_r50.jsonl / LEADERBOARD_r50.md.

Reuses the arch-agnostic custom directions in score_experiments.py for --phase experiments
(they go through ctx; ctx.mtype routes the iterative model_type).

Usage:
    python harness_r50.py --phase baseline
    python harness_r50.py --phase experiments
    python harness_r50.py --only prop_cov --val_limit 10000
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
import torch
import torchvision.models as tv_models
import torch_pruning as tp
from PIL import Image

from vbp_common import get_val_transform
from normalize_net import build_whole_net_reparam_layers, build_reparam_manager
from normalized_net_importance import extract_normnet_scores, NormalizedNetImportance
from torch_pruning.utils.reparam import fold_all_conv_bn
import normnet_main as NM

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/Users/avrahamraviv/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
DATA = os.path.join(HERE, "..", "data", "imagenet1k_val", "data")
RESULTS = os.path.join(HERE, "results_r50.jsonl")
LEADERBOARD = os.path.join(HERE, "LEADERBOARD_r50.md")
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
    """In-memory (x,y) batches — cached calib set, reused for every stats collector."""
    def __init__(self, batches): self.batches = batches
    def __iter__(self): return iter(self.batches)
    def __len__(self): return len(self.batches)


@torch.no_grad()
def recalibrate_bn(model, calib, device, max_batches):
    """Reset + re-estimate BN running stats on calib after pruning. Pruning a producer's output
    channels changes every downstream BN's input distribution → stale running mean/var collapse
    accuracy. This is the pre-FT analogue of normnet's 'reinsert fresh BN at pruned widths'.
    No-op on BN-free nets."""
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
        # dense model (original, un-normalized, WITH native BN) for classical scorers + base
        self.dense = tv_models.resnet50(weights=None)
        sd = torch.load(CKPT, map_location="cpu")
        self.dense.load_state_dict(sd.get("model", sd), strict=False)
        self.dense.eval()
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
        # resnet has native BN → fold Conv->BN into the conv on the scoring copy BEFORE reparam,
        # so v_tilde=σW and σ_out are measured POST-BN (function-preserving). dense stays unfolded
        # (classical scorers + pruning base score the original).
        self.norm_model = copy.deepcopy(self.dense)
        n_fold, _ = fold_all_conv_bn(self.norm_model)
        print(f"[ctx] folded {n_fold} Conv->BN on scoring copy", flush=True)
        nargs = SimpleNamespace(reparam_variant="mean", norm_bn_momentum=0.01,
                                mu_ema_momentum=0.0, calib_batches=self.args.calib_batches,
                                max_batches=self.args.calib_batches, model_type=MTYPE,
                                cnn_arch="resnet50", exclude_stem=False)
        names = build_whole_net_reparam_layers(self.norm_model, exclude_classifier=True,
                                               exclude_stem=False)
        self.mgr = build_reparam_manager(self.norm_model, names, "cpu", nargs)
        print("[ctx] calibrating reparam σ/μ...", flush=True)
        self.mgr.reparameterize(self.calib)
        self.clf = NM._classifier(self.norm_model)
        self.bscale = NM._propagation_branch_scale(self.norm_model)   # None for resnet
        self.rblocks = NM._residual_blocks(self.norm_model)
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
        if self._jcov is None:
            print("[ctx] collecting join covariance...", flush=True)
            self._jcov = self.mgr.collect_join_covariance(self.calib, self.rblocks,
                                                          max_batches=self.args.calib_batches)
        return self._jcov

    def extract_kwargs(self, cov=False, join_cov=False, p=2, relative=True,
                       measured_var=False):
        return dict(p=p, relative=relative, classifier=self.clf, use_measured_sigma_c=False,
                    use_measured_var=measured_var, branch_out_scale=self.bscale,
                    input_cov=(self.icov if cov else None),
                    join_cov=(self.jcov if join_cov else None))

    def base(self):
        """Pruning base = ORIGINAL dense (with BN). Scores map to modules by NAME; the folded
        scoring copy and the original share conv names + channel indices (fold/reparam only
        rescale weights, don't restructure), so pruning the original at those indices is the
        equivalent pre-FT base for every spec. TP prunes conv+BN together."""
        return copy.deepcopy(self.dense)

    def var_stats(self, mode):
        if self._var_stats is None:
            tl = NM._post_act_target_layers(self.dense, MTYPE, self.ex)
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
            tl = NM._post_act_target_layers(self.dense, MTYPE, self.ex)
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
                normalizer=_norm_arg(norm)), ctx.base(), norm)
    if kind == "per_layer":
        scores = ctx.mgr.input_channel_scores()
        return (lambda mdl: NormalizedNetImportance(mdl, scores, group_reduction="mean",
                normalizer=_norm_arg(norm)), ctx.base(), norm)
    if kind == "propagation":
        ek = ctx.extract_kwargs(cov=p.get("cov", False), join_cov=p.get("join_cov", False),
                                p=p.get("p", 2), relative=p.get("relative", True),
                                measured_var=p.get("measured_var", False))
        scores = extract_normnet_scores(ctx.mgr, "propagation", example_inputs=ctx.ex, **ek)
        return (lambda mdl: NormalizedNetImportance(mdl, scores, group_reduction="mean",
                normalizer=_norm_arg(norm)), ctx.base(), norm)
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
                normalizer=None), ctx.base(), "none")
    if kind == "custom":
        import score_experiments as SE
        fn = getattr(SE, p["fn"])
        scores = fn(ctx, p)
        cn = p.get("prune_normalizer", norm)
        return (lambda mdl: NormalizedNetImportance(mdl, scores, group_reduction="mean",
                normalizer=_norm_arg(cn)), ctx.base(), cn)
    raise ValueError(f"unknown kind {kind!r}")


def run_spec(ctx, spec, limit):
    imp_factory, base, _norm = build_importance_factory(ctx, spec)
    model = copy.deepcopy(base)
    ignored = lambda mdl: NM._ignored_layers(mdl, MTYPE)
    ratio = ctx.args.pruning_ratio
    if ctx.args.mac_target_g > 0:
        ratio = NM._ratio_for_mac(model, ctx.ex, imp_factory, ignored,
                                  ctx.args.mac_target_g, global_pruning=True)
    mean_dict = None
    if ctx.args.bias_comp:
        mbn = ctx.means_by_name()
        mean_dict = {m: mbn[nm] for nm, m in model.named_modules() if nm in mbn}
    tp.pruner.MagnitudePruner(model, ctx.ex, importance=imp_factory(model),
                              global_pruning=True, pruning_ratio=ratio,
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
        f.write("# Retention leaderboard — ResNet-50, pre-FT top-1 @ matched MAC\n\n")
        f.write("| rank | spec | top-1 | MAC% | params(M) | n | normalizer | notes |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(rows, 1):
            res, sp = r["res"], r["spec"]
            f.write(f"| {i} | `{sp['name']}` | {res['acc']:.4f} | {res['mac_pct']:.0f}% | "
                    f"{res['params']:.2f} | {res['n']} | "
                    f"{sp.get('params', {}).get('normalizer', '-')} | {sp.get('note', '')} |\n")


# --------------------------------------------------------------------- specs
def baseline_specs(normalizer="width"):
    P = lambda **k: dict(normalizer=normalizer, **k)
    return [
        dict(name="random", kind="random"),
        dict(name="magnitude", kind="magnitude", params=P()),
        dict(name="variance_vbp", kind="variance", params=P(), note="VBP activation σ²"),
        dict(name="tp_variance", kind="tp_variance", params=P(), note="group-L2×σ"),
        dict(name="nci_cov", kind="nci_cov", params=P()),
        dict(name="per_layer_nci", kind="per_layer", params=P(), note="‖σv‖"),
        dict(name="prop_rel_p2", kind="propagation", params=P(relative=True, p=2)),
        dict(name="prop_nonrel_p1", kind="propagation", params=P(relative=False, p=1),
             note="std-prop diagnostic"),
        dict(name="prop_p2_cov", kind="propagation", params=P(cov=True)),
        dict(name="prop_p2_joincov", kind="propagation", params=P(join_cov=True)),
        dict(name="prop_p2_cov_joincov", kind="propagation", params=P(cov=True, join_cov=True)),
        dict(name="prop_p2_cov_joincov_normNone", kind="propagation",
             params=dict(normalizer="none", cov=True, join_cov=True)),
        dict(name="prop_p2_cov_joincov_normMean", kind="propagation",
             params=dict(normalizer="mean", cov=True, join_cov=True)),
        dict(name="prop_iter_cov_joincov", kind="propagation_iterative",
             params=P(cov=True, join_cov=True, iter_drop=128, iter_max_frac=0.6)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="baseline", choices=["baseline", "experiments"])
    ap.add_argument("--only", default="", help="run only specs whose name contains this")
    ap.add_argument("--eval_device", default="mps")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--calib_batches", type=int, default=50)
    ap.add_argument("--val_limit", type=int, default=5000)
    ap.add_argument("--confirm_limit", type=int, default=10000)
    ap.add_argument("--mac_target_g", type=float, default=0.0,
                    help="target GMAC. <=0 ⇒ auto = 0.66×dense (mirror convnext budget).")
    ap.add_argument("--pruning_ratio", type=float, default=0.5)
    ap.add_argument("--normalizer", default="width")
    ap.add_argument("--no_bias_comp", action="store_true")
    ap.add_argument("--no_bn_recalib", action="store_true",
                    help="skip post-prune BN re-estimation (default ON for resnet — required "
                         "for meaningful pre-FT on a BN net).")
    args = ap.parse_args()
    args.bias_comp = not args.no_bias_comp

    if args.phase == "baseline":
        specs = baseline_specs(args.normalizer)
    else:
        import score_experiments as SE
        specs = SE.SPECS
    if args.only:
        specs = [s for s in specs if args.only in s["name"]]

    ctx = Ctx(args)
    if args.mac_target_g <= 0:
        args.mac_target_g = round(0.66 * ctx.dense_macs, 2)
        print(f"[harness] auto MAC target = {args.mac_target_g}G (0.66×dense)", flush=True)
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
