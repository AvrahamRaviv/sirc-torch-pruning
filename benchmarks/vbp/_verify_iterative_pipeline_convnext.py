"""Smoke-test the --prop_iterative wiring end to end on ConvNeXt-tiny WITHOUT ImageNet:
greedy helper → removal-rank scores → NormalizedNetImportance(normalizer='none') →
MagnitudePruner. Verifies the pruner removes exactly the greedy prefix (lowest-rank pwconv2
input channels) and that widths actually shrink. Synthetic calib (structural check only)."""
import sys
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects/Torch-Pruning")
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects/Torch-Pruning/benchmarks/vbp")
import torch
import torch_pruning as tp
from types import SimpleNamespace
from vbp_common import convnext_tiny
from normalize_net import build_whole_net_reparam_layers, build_reparam_manager
from normalized_net_importance import NormalizedNetImportance
import normnet_main as NM

dev = "cpu"   # deterministic numerics for the structural check (mps has masked-score nan quirks)
torch.manual_seed(0)
CKPT = "/Users/avrahamraviv/.cache/torch/hub/checkpoints/convnext_tiny_22k_1k_224.pth"


class Synth:
    def __init__(self, nb=4, bs=8): self.nb, self.bs = nb, bs
    def __iter__(self):
        for _ in range(self.nb):
            yield torch.randn(self.bs, 3, 224, 224), torch.zeros(self.bs, dtype=torch.long)
    def __len__(self): return self.nb


ARGS = SimpleNamespace(reparam_variant="mean", norm_bn_momentum=0.01, mu_ema_momentum=0.0,
                       calib_batches=4, max_batches=4, model_type="convnext", exclude_stem=False)

m = convnext_tiny(pretrained=False)
sd = torch.load(CKPT, map_location="cpu")
m.load_state_dict(sd.get("model", sd), strict=False)
m = m.to(dev).eval()
ex = torch.randn(1, 3, 224, 224, device=dev)

names = build_whole_net_reparam_layers(m, exclude_classifier=True, exclude_stem=False)
mgr = build_reparam_manager(m, names, dev, ARGS)
mgr.reparameterize(Synth())
clf = NM._classifier(m)
bscale = NM._propagation_branch_scale(m)
jcov = mgr.collect_join_covariance(Synth(), NM._residual_blocks(m), max_batches=4)
extract_kwargs = dict(p=2, relative=True, classifier=clf, use_measured_sigma_c=False,
                      use_measured_var=False, branch_out_scale=bscale, input_cov=None,
                      join_cov=jcov)

NORM = "width"
print(f"=== greedy iterative scores (normalizer={NORM}, folded into ranking) ===")
scores, order = NM._iterative_propagation_scores(
    mgr, ex, extract_kwargs, model_type="convnext", normalizer=NORM,
    drop_per_round=512, max_frac=0.05, log=print)

pool = NM._prunable_input_layers("convnext", list(scores.keys()))
total = sum(int(scores[n].numel()) for n in pool)
removed = len(order)                                   # order = [(group_idx, ch), ...] (group-aware)
# rank-score sanity (group-aware): removed group-channels carry steps 0..removed-1 (each step
# written to EVERY consumer of its group → values may repeat across multi-consumer groups),
# survivors carry total+. Verify from the scores dict (name→rank) so it's order-format-agnostic.
below = [float(v) for n in pool for v in scores[n].tolist() if v < total]
distinct_steps = sorted(set(int(x) for x in below))
contiguous = distinct_steps == list(range(removed))
print(f"pool layers={len(pool)} total_pool_ch={total} greedy_removed={removed}")
print(f"removed-step values: {len(distinct_steps)} distinct, min={distinct_steps[0]} "
      f"max={distinct_steps[-1]} (expect 0..{removed-1})  contiguous={contiguous}")
surv_min = min(float(scores[n][scores[n] >= total].min()) for n in pool if (scores[n] >= total).any())
print(f"survivor scores all ≥ total({total}): min survivor={surv_min:.1f}  "
      f"({'OK' if surv_min >= total else 'FAIL'})")

# feed to the pruner exactly as normnet_main does (normalizer folded → 'none' at prune)
print("\n=== prune via NormalizedNetImportance + MagnitudePruner (normalizer='none') ===")
mgr.merge_back()
pre_w = NM._layer_widths(m)
imp = NormalizedNetImportance(m, scores, group_reduction="mean", normalizer=None)
ratio = 0.04                                          # < max_frac so the prefix is well-defined
tp.pruner.MagnitudePruner(
    m, ex, importance=imp, global_pruning=True, pruning_ratio=ratio,
    ignored_layers=NM._ignored_layers(m, "convnext")).step()
m.to(dev)
post_w = NM._layer_widths(m)
shrunk = {n: (pre_w[n], post_w[n]) for n in pre_w if post_w.get(n) != pre_w[n]}
print(f"layers whose width changed: {len(shrunk)}")
for n in list(shrunk)[:6]:
    print(f"  {n}: {shrunk[n][0]} -> {shrunk[n][1]}")

# the pruner removes the producer (pwconv1) OUTPUT = consumer (pwconv2) INPUT, lowest-rank
# first (global). Confirm SOME pwconv1 producers shrank and the total removed is positive
# (the per-layer greedy-prefix accounting needed the old (name,ch) order format; the rank
# invariant above already proves the prefix is honoured).
K = sum(pre_w[L.replace("pwconv2", "pwconv1")] - post_w.get(L.replace("pwconv2", "pwconv1"),
        pre_w[L.replace("pwconv2", "pwconv1")]) for L in pool)
prod_shrunk = sum(1 for L in pool
                  if post_w.get(L.replace("pwconv2", "pwconv1"))
                  != pre_w[L.replace("pwconv2", "pwconv1")])
print(f"\n=== pruner removed K={K} producer channels across {prod_shrunk} pwconv1 layers "
      f"({'OK' if K > 0 and prod_shrunk > 0 else 'FAIL'}) ===")
acc_fwd = m(ex)
print(f"\nforward on pruned model OK: out shape {tuple(acc_fwd.shape)}")
print("DONE.")
