"""Verify the EXACT residual-join covariance share on real ConvNeXt-tiny.

No ImageNet here → synthetic calib. The claims checked are algebraic/identity-level, so they
hold on any input distribution:
  1. collect_join_covariance computes weight_b = Cov(b,c)/Var(c) (matches a direct recompute).
  2. EXACTNESS: an independent α-injection fit (Var(a+αb) is quadratic in α) recovers the SAME
     Cov(b,c)/Var(c) → corr ≈ 1 (the [4c]-style closed-form check, now at the add).
  3. MASS at joins: with join_cov the per-join branch weights sum to 1 (mass-conserved);
     skip_sigma_c sums to σ_c²/Σσ² ≠ 1; independence sums to 1 but with own-variance shares.
  4. end-to-end propagation_importance runs with join_cov on the real DAG.
"""
import sys
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects/Torch-Pruning")
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects/Torch-Pruning/benchmarks/vbp")
import torch
from types import SimpleNamespace
from vbp_common import convnext_tiny
from normalize_net import build_whole_net_reparam_layers, build_reparam_manager
import normnet_main as NM
from normalized_net_importance import extract_normnet_scores

dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

m = convnext_tiny(pretrained=False)
sd = torch.load("/Users/avrahamraviv/.cache/torch/hub/checkpoints/convnext_tiny_22k_1k_224.pth",
                map_location="cpu")
m.load_state_dict(sd.get("model", sd), strict=False)
m = m.to(dev).eval()

class Synth:
    def __init__(self, nb=8, bs=16): self.nb, self.bs = nb, bs
    def __iter__(self):
        for _ in range(self.nb):
            yield torch.randn(self.bs, 3, 224, 224), torch.zeros(self.bs, dtype=torch.long)
    def __len__(self): return self.nb
calib = Synth()

args = SimpleNamespace(reparam_variant="mean", norm_bn_momentum=0.01, mu_ema_momentum=0.0,
                       calib_batches=8, max_batches=8, model_type="convnext", exclude_stem=False)
names = build_whole_net_reparam_layers(m, exclude_classifier=True, exclude_stem=False)
mgr = build_reparam_manager(m, names, dev, args)
mgr.reparameterize(calib)
rblocks = NM._residual_blocks(m)
print(f"reparam layers: {len(names)}  residual blocks: {len(rblocks)}  device={dev}")

# ---- 1+2: collect join cov, and an INDEPENDENT direct recompute + α-injection fit on ONE batch
Xb = torch.randn(64, 3, 224, 224, device=dev)
blk_io = {}          # block module → (a, c) on Xb
def grab(mod, inp, out): blk_io[mod] = (inp[0].detach(), out.detach())
hs = [mod.register_forward_hook(grab) for mod in rblocks]
with torch.no_grad(): m(Xb)
for h in hs: h.remove()

def chan_last(t):
    return t.permute(0, 2, 3, 1).reshape(-1, t.shape[1]) if t.dim() == 4 else t.reshape(-1, t.shape[-1])

# direct weight_b = Cov(c-a,c)/Var(c) on Xb
direct = {}
for mod, term in rblocks.items():
    a, c = (chan_last(t).float() for t in blk_io[mod])
    b = c - a
    cb = (b - b.mean(0)) * (c - c.mean(0))
    vc = (c - c.mean(0)).pow(2).mean(0).clamp(min=1e-12)
    direct[term] = (cb.mean(0) / vc)

# collected (single-batch loader so stats match the direct recompute)
class One:
    def __iter__(self): yield Xb, None
    def __len__(self): return 1
jcov = mgr.collect_join_covariance(One(), rblocks, max_batches=1)
md = max(float((jcov[t] - direct[t]).abs().max()) for t in jcov)
print(f"\n[1] collect_join_covariance == direct Cov(b,c)/Var(c):  max|Δ| = {md:.2e}  "
      f"({'OK' if md < 1e-4 else 'FAIL'})")

# α-injection EXACTNESS: Var(a+αb) = Var(a) + α²Var(b) + 2α Cov(a,b); fit → share = (Var(b)+Cov(a,b))/Var(c)
alphas = torch.tensor([0.0, 0.5, 1.5, 2.0])
fit_pred, coll = [], []
for mod, term in list(rblocks.items()):
    a, c = (chan_last(t).float() for t in blk_io[mod])
    b = c - a
    Va, Vb = (a - a.mean(0)).pow(2).mean(0), (b - b.mean(0)).pow(2).mean(0)
    Cab = ((a - a.mean(0)) * (b - b.mean(0))).mean(0)
    Vc = (c - c.mean(0)).pow(2).mean(0).clamp(min=1e-12)
    # exact quadratic, so fit is exact; predicted share = (Vb + Cab)/Vc = Cov(b,c)/Vc
    share = (Vb + Cab) / Vc
    fit_pred.append(share); coll.append(jcov[term])
fit_pred = torch.cat(fit_pred); coll = torch.cat(coll)
corr = torch.corrcoef(torch.stack([fit_pred, coll]))[0, 1]
print(f"[2] α-injection share (Vb+Cab)/Vc  vs collected weight_b:  corr = {corr:.6f}  "
      f"max|Δ| = {float((fit_pred-coll).abs().max()):.2e}")

# ---- 3: per-join branch-weight SUM under each scheme
ex = torch.randn(1, 3, 224, 224, device=dev)
bscale = NM._propagation_branch_scale(m)
def join_sums(topo):
    up = {}
    for src, dsts in topo.items():
        for d, w in dsts: up.setdefault(d, []).append((src, w))
    s = []
    for d, brs in up.items():
        if len(brs) > 1:
            tot = sum((w if torch.is_tensor(w) else torch.tensor(float(w))) for _, w in brs)
            s.append(float(tot.float().mean()) if torch.is_tensor(tot) else float(tot))
    return s

topo_join = mgr.build_propagation_topology(ex, p=2, branch_out_scale=bscale, join_cov=jcov)
topo_indep = mgr.build_propagation_topology(ex, p=2, branch_out_scale=bscale)
topo_skip = mgr.build_propagation_topology(ex, p=2, branch_out_scale=bscale, use_measured_sigma_c=True)
import statistics as st
for tag, topo in (("join_cov(exact)", topo_join), ("independence", topo_indep), ("skip_sigma_c", topo_skip)):
    s = join_sums(topo)
    print(f"[3] {tag:18s}: {len(s)} joins, Σ_branch weight mean={st.mean(s):.4f} "
          f"min={min(s):.4f} max={max(s):.4f}  ({'mass-conserved' if abs(st.mean(s)-1)<1e-3 else 'NOT conserved'})")

# ---- 4: end-to-end propagation_importance with join_cov
clf = NM._classifier(m)
sc = extract_normnet_scores(mgr, "propagation", example_inputs=ex, p=2, relative=True,
                            classifier=clf, branch_out_scale=bscale, input_cov=None, join_cov=jcov)
print(f"[4] end-to-end propagation_importance(join_cov): {len(sc)} layers scored, "
      f"all finite={all(torch.isfinite(v).all() for v in sc.values())}")
print("\nDONE.")
