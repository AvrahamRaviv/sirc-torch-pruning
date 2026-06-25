"""DeiT-tiny autoresearch retention harness — sibling of harness.py / harness_mbv2.py.

PURE pre-FT. DeiT has NO native BN (LayerNorm only) → nothing to fold, no stale stats,
no recalibration. Every config scores+prunes the SAME dense timm deit_tiny and is evaluated
as-is. Fair pre-FT comparison.

Scope = MLP-only, exactly mirroring the convnext arm: prune ONLY each block's mlp.fc1 output
(the 4×dim hidden, = mlp.fc2 input). Attention (qkv/proj), patch_embed, head, the residual
stream (embed dim) and all norms are IGNORED — that keeps TP's group graph trivial (fc1→fc2)
and avoids head-dim / residual-width coupling. This is the same prunable dim the convnext
harness uses, so propagation scores transfer 1:1.

timm deit_tiny_patch16_224 (fb_in1k), model_type='vit'. _classifier → .head (timm). Local
DeiT target-layer + ignored-layer shims (the pushed normnet_main vit path hardcodes HF
`model.vit.encoder.layer`, which timm DeiT does not have). join_cov off (no residual blocks
in MLP-only scope). Own results_deit.jsonl / LEADERBOARD_deit.md. Reuses score_experiments.py.

Usage:
    python harness_deit.py --phase baseline
    python harness_deit.py --phase experiments
    python harness_deit.py --only prop_cov --val_limit 10000
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
import torch_pruning as tp
from PIL import Image

from vbp_common import get_val_transform
from normalize_net import build_whole_net_reparam_layers, build_reparam_manager
from normalized_net_importance import extract_normnet_scores, NormalizedNetImportance
import normnet_main as NM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "imagenet1k_val", "data")
RESULTS = os.path.join(HERE, "results_deit.jsonl")
LEADERBOARD = os.path.join(HERE, "LEADERBOARD_deit.md")
MTYPE = "vit"


# ----------------------------------------------------------------- DeiT adapters (local)
def deit_target_layers(model):
    """(producer, post_act_fn) per block for stats: mlp.fc1 + its GELU (the pruned dim)."""
    return [(blk.mlp.fc1, blk.mlp.act) for blk in model.blocks]


def deit_ignored(model):
    """MLP-only: ignore everything except each block's mlp.fc1 → leaves only fc1.out prunable
    (fc2.in follows by TP coupling). Mirrors convnext _ignored (pwconv1 only prunable)."""
    ig = [model.head, model.patch_embed.proj]
    for blk in model.blocks:
        ig += [blk.attn.qkv, blk.attn.proj, blk.mlp.fc2]
    return ig


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
        m = timm.create_model("deit_tiny_patch16_224", pretrained=True)
        m.eval()
        print("[ctx] loaded timm deit_tiny_patch16_224 (fb_in1k)", flush=True)
        self.dense = m
        self.dense_macs, self.dense_params = NM._count(self.dense, self.ex)
        print(f"[ctx] decoding {args.calib_batches} calib batches...", flush=True)
        self.calib = ListLoader(list(_batched(
            _rows(self.shards[::-1], limit=args.calib_batches * args.batch_size),
            self.tf, args.batch_size)))
        self._var_stats = None
        self._nci_scores = None
        self._means_by_name = None
        self._build_normnet()

    def _build_normnet(self):
        self.norm_model = copy.deepcopy(self.dense)
        nargs = SimpleNamespace(reparam_variant="mean", norm_bn_momentum=0.01,
                                mu_ema_momentum=0.0, calib_batches=self.args.calib_batches,
                                max_batches=self.args.calib_batches, model_type=MTYPE,
                                cnn_arch="deit_tiny", exclude_stem=False)
        names = build_whole_net_reparam_layers(self.norm_model, exclude_classifier=True,
                                               exclude_stem=False)
        # MLP-only reparam: drop attention (qkv/proj) + patch_embed from the propagation graph.
        # Attention's qkv(576=3×192)→reshape→softmax→proj(192) is NOT a linear chain, so the
        # propagation DAG mis-chains qkv→proj (576≠192 shape mismatch). We only prune mlp.fc1
        # anyway → score only the mlp chain (fc1→fc2) through the residual stream.
        names = [n for n in names if ".mlp." in n]
        try:
            self.mgr = build_reparam_manager(self.norm_model, names, "cpu", nargs)
            print("[ctx] calibrating reparam σ/μ...", flush=True)
            self.mgr.reparameterize(self.calib)
        except Exception as e:
            print(f"[ctx] WARN reparam failed ({e}); propagation specs will error, "
                  f"baselines still run", flush=True)
            self.mgr = None
        self.clf = NM._classifier(self.norm_model)         # timm .head → found
        self.bscale = None
        self.rblocks = {}                                  # MLP-only: no residual join
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
        return {}

    def extract_kwargs(self, cov=False, join_cov=False, p=2, relative=True,
                       measured_var=False):
        return dict(p=p, relative=relative, classifier=self.clf, use_measured_sigma_c=False,
                    use_measured_var=measured_var, branch_out_scale=self.bscale,
                    input_cov=(self.icov if cov else None), join_cov=None)

    def base(self):
        return copy.deepcopy(self.dense)

    def var_stats(self, mode):
        if self._var_stats is None:
            tl = deit_target_layers(self.dense)
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
            tl = deit_target_layers(self.dense)
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
        _ig_ids = {id(m) for m in deit_ignored(ctx.norm_model)}
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
    ignored = lambda mdl: deit_ignored(mdl)
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
        f.write("# Retention leaderboard — DeiT-tiny (MLP-only), PURE pre-FT @ matched MAC\n\n")
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
        dict(name="prop_p2_cov", kind="propagation", params=P(cov=True)),
        dict(name="prop_p2_cov_normMean", kind="propagation",
             params=dict(normalizer="mean", cov=True)),
    ]


def vitjoin_specs():
    """Residual-join DAG experiments: measured global stream share g_i for cross-block budget."""
    return [
        dict(name="prop_vit_joinshare", kind="custom",
             params=dict(fn="prop_vit_joinshare", cov=True, measured_var=True,
                         prune_normalizer="none"),
             note="cov+measured within-MLP × measured stream share g_i"),
        dict(name="prop_vit_joinshare_plain", kind="custom",
             params=dict(fn="prop_vit_joinshare", cov=False, measured_var=False,
                         prune_normalizer="none"),
             note="plain prop within-MLP × g_i (cov off)"),
        dict(name="prop_vit_joinshare_meas", kind="custom",
             params=dict(fn="prop_vit_joinshare", cov=False, measured_var=True,
                         prune_normalizer="none"),
             note="measured-only within-MLP × g_i"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="baseline",
                    choices=["baseline", "experiments", "vitjoin"])
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
    args = ap.parse_args()
    args.bias_comp = not args.no_bias_comp

    if args.phase == "baseline":
        specs = baseline_specs(args.normalizer)
    elif args.phase == "vitjoin":
        specs = vitjoin_specs()
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
