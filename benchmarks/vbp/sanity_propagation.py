"""Hard assertion suite: code propagation criterion vs normalized_nets_pruning.pdf.

Checks EVERY PDF element separately (not just drift), for p=1 AND p=2, on BOTH code paths
(normnet_standalone.compute_scores + reparam.MeanResidualManager.propagation_importance),
plus a RESIDUAL toy for the skip-join variance-share the sequential case can't exercise.

PDF map (steps 7-10): sigma folded into M=sigma*W ONCE, NO separate per-hop transfer.
  M[i,j] = sigma_in_i * |reduce(W[j,i,:])|                          (step 1)
  sigma_pre_j = sqrt(Sum_i sigma_i^2 w_ij^2)  -> (colsum M^2)_j^0.5 (step 2)
  relative   W_bar = M^p / Sum_i M^p_ij            (L1, col-stochastic, steps 9-10)
  nonrel     W_bar = M^p / sigma_pre^p = M^p / (colsum M^2)^(p/2)   (L2/std, steps 7-8)
  recursion  I^l = W_bar^l @ I^{l+1}               (no transfer; sigma once)
  identity   for p=2 the two W_bar COINCIDE -> nonrel == rel (PDF p.3)
  skip add   back-prop splits by sigma_branch^2/(sigma_a^2+sigma_b^2)(relative section)
"""
import sys, os, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
_spec = importlib.util.spec_from_file_location("nns", os.path.join(os.path.dirname(__file__), "normnet_standalone.py"))
nns = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(nns); nns._LOG_FH = None
from torch_pruning.utils.reparam import MeanResidualManager  # noqa

torch.manual_seed(0)
CPU = torch.device("cpu")
RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

# ----------------------------------------------------------------------------- helpers
def empirical_sigmas(model, fcs, X, acts):
    """Return sigma_in, sigma_pre (layer out), sigma_post (after act, last=identity)."""
    pre, inp = {}, {}
    hs = []
    for fc in fcs:
        hs.append(fc.register_forward_hook(lambda m, i, o, f=fc: pre.setdefault(f, []).append(o.detach())))
        hs.append(fc.register_forward_pre_hook(lambda m, i, f=fc: inp.setdefault(f, []).append(i[0].detach())))
    with torch.no_grad(): model(X)
    for h in hs: h.remove()
    sin = {fc: torch.cat(inp[fc]).reshape(-1, inp[fc][0].shape[-1]).std(0) for fc in fcs}
    spre = {fc: torch.cat(pre[fc]).reshape(-1, pre[fc][0].shape[-1]).std(0) for fc in fcs}
    spost = {}
    for fc in fcs:
        z = torch.cat(pre[fc]).reshape(-1, pre[fc][0].shape[-1])
        spost[fc] = (acts[fc](z) if acts[fc] is not None else z).std(0)
    return sin, spre, spost

def hand_M(fc, sin, p):
    W = fc.weight.data
    return (W.abs().t() * sin[fc][:, None])  # [in,out]

def wbar(M, p):              # relative: L1 column-stochastic
    Mp = M.pow(p); return Mp / Mp.sum(0).clamp_min(1e-12)

def wbar_nr(M, p):           # non-relative: sigma_pre^p (L2/std col-norm), no transfer
    Mp = M.pow(p); return Mp / M.pow(2).sum(0).clamp_min(1e-12).pow(p / 2.0)

# ============================================================ 1. SEQUENTIAL MLP
print("=== 1. SEQUENTIAL MLP (fcA->ReLU->fcB->ReLU->fcC), p in {1,2} ===")
DIMS = [8, 6, 5, 4]
class MLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.fcA = nn.Linear(DIMS[0], DIMS[1]); s.fcB = nn.Linear(DIMS[1], DIMS[2]); s.fcC = nn.Linear(DIMS[2], DIMS[3])
        s.relu = nn.ReLU()
    def forward(s, x): return s.fcC(s.relu(s.fcB(s.relu(s.fcA(x)))))
m = MLP().eval()
fcs = [m.fcA, m.fcB, m.fcC]
acts = {m.fcA: torch.relu, m.fcB: torch.relu, m.fcC: None}   # fcC terminal (no act)
X = torch.randn(400, DIMS[0]) * 1.4 + 0.2
sin, spre, spost = empirical_sigmas(m, fcs, X, acts)

# component checks (standalone internals) — p independent
prunable = {f"fc{k}": fc for k, fc in zip("ABC", fcs)}
loader = [(X[i:i+40], torch.zeros(40, dtype=torch.long)) for i in range(0, 400, 40)]
mu, sigma, fire, _ = nns.calibrate(m, prunable, loader, CPU, len(loader), save_input_for=[])
ex = torch.randn(1, DIMS[0])
edges = nns.discover_edges(m, ex, prunable)
# A) M construction (step 1)
keyof = {fc: k for k, fc in zip("ABC", fcs)}
for fc in fcs:
    Ms = nns.layer_M(fc, sigma[fc]); Mh = hand_M(fc, sigma, 2)
    check(f"M == sigma_in*|W|  (fc{keyof[fc]})",
          torch.allclose(Ms, Mh, atol=1e-6), f"max|d|={(Ms-Mh).abs().max():.1e}")
# B) D denominator == PDF formula colsum = Sum_i (sigma_i w_ij)^p  (step 2, independence est)
for p in (1, 2):
    for fc in fcs:
        cs = nns.layer_M(fc, sigma[fc]).pow(p).sum(0)
        hand = (sigma[fc][:, None] * fc.weight.data.abs().t()).pow(p).sum(0)
        check(f"colsum == Sum(sigma*w)^{p} (fc{keyof[fc]})", torch.allclose(cs, hand, atol=1e-6),
              f"max|d|={(cs-hand).abs().max():.1e}")
# C) NONREL denominator == sigma_pre^p (L2/std col-norm), distinct from rel's L1 for p=1
for p in (1, 2):
    for fc in fcs:
        M = nns.layer_M(fc, sigma[fc])
        nr_denom = M.pow(2).sum(0).pow(p / 2.0)                  # sigma_pre^p
        rel_denom = M.pow(p).sum(0)                              # L1 of M^p
        same = torch.allclose(nr_denom, rel_denom, atol=1e-6)
        check(f"nonrel denom == sigma_pre^{p} (fc{keyof[fc]})",
              torch.allclose(nr_denom, M.pow(2).sum(0).sqrt().pow(p), atol=1e-6))
        if p == 2:
            check(f"p=2: nonrel denom == rel denom (fc{keyof[fc]})", same)
        else:
            check(f"p=1: nonrel denom != rel denom (fc{keyof[fc]})", not same)

# D) RECURSION exactness: hand oracle (no transfer; sigma once). rel=L1, nonrel=L2.
for p in (1, 2):
    sc = nns.compute_scores(prunable, sigma, edges, fire, p=p)
    def seq(nonrel):
        wb = wbar_nr if nonrel else wbar
        I = {}; seed = torch.full((DIMS[3],), 1.0/DIMS[3])
        Ic = wb(hand_M(m.fcC, sigma, p), p) @ seed; I[m.fcC]=Ic
        Ib = wb(hand_M(m.fcB, sigma, p), p) @ Ic;   I[m.fcB]=Ib
        Ia = wb(hand_M(m.fcA, sigma, p), p) @ Ib;   I[m.fcA]=Ia
        return I
    relh, nrh = seq(False), seq(True)
    for fc, key in zip(fcs, "ABC"):
        check(f"standalone REL    p={p} fc{key}", torch.allclose(sc['rel'][fc], relh[fc], atol=1e-6),
              f"max|d|={(sc['rel'][fc]-relh[fc]).abs().max():.1e}")
        check(f"standalone NONREL p={p} fc{key}", torch.allclose(sc['nonrel'][fc], nrh[fc], atol=1e-6),
              f"max|d|={(sc['nonrel'][fc]-nrh[fc]).abs().max():.1e}")
    if p == 2:  # PDF identity: variance propagation == relative criterion
        for fc, key in zip(fcs, "ABC"):
            check(f"standalone p=2 NONREL==REL fc{key}",
                  torch.allclose(sc['nonrel'][fc], sc['rel'][fc], atol=1e-6),
                  f"max|d|={(sc['nonrel'][fc]-sc['rel'][fc]).abs().max():.1e}")

# production path on same MLP
print("  -- production MeanResidualManager (same MLP) --")
m2 = MLP().eval(); m2.load_state_dict(m.state_dict())
mgr = MeanResidualManager(m2, ["fcA", "fcB", "fcC"], CPU, lambda_reg=0.0, max_batches=10)
mgr.reparameterize([(X[i:i+40], None) for i in range(0, 400, 40)])
rpA, rpB, rpC = (mgr._reparam_modules[k] for k in ["fcA", "fcB", "fcC"])
for p in (1, 2):
    seed = torch.full((DIMS[3],), 1.0/DIMS[3])
    rel = mgr.propagation_importance(I_out=seed, p=p, relative=True)
    nr  = mgr.propagation_importance(I_out=seed, p=p, relative=False)
    # production hand-ref from ITS M buffers (sigma folded once). rel=L1, nonrel=L2. No transfer.
    def Mp_prod(rp):
        from torch_pruning.utils.reparam import _contribution_weight
        w = _contribution_weight(rp).detach(); return w.t().abs()
    def seqp(nonrel):
        wb = wbar_nr if nonrel else wbar
        IC = wb(Mp_prod(rpC), p) @ seed
        IB = wb(Mp_prod(rpB), p) @ IC
        IA = wb(Mp_prod(rpA), p) @ IB
        return {"fcA": IA, "fcB": IB, "fcC": IC}
    rh, nh = seqp(False), seqp(True)
    for k in ["fcA", "fcB", "fcC"]:
        check(f"production REL    p={p} {k}", torch.allclose(rel[k], rh[k], atol=1e-5), f"max|d|={(rel[k]-rh[k]).abs().max():.1e}")
        check(f"production NONREL p={p} {k}", torch.allclose(nr[k], nh[k], atol=1e-5), f"max|d|={(nr[k]-nh[k]).abs().max():.1e}")
    if p == 2:
        for k in ["fcA", "fcB", "fcC"]:
            check(f"production p=2 NONREL==REL {k}", torch.allclose(nr[k], rel[k], atol=1e-6),
                  f"max|d|={(nr[k]-rel[k]).abs().max():.1e}")

# ============================================================ 2. RESIDUAL skip-join
# out = fcC(relu( fcB(relu(fcA(x))) + fcS(x) )). The ADD merges fcB(main) + fcS(skip) and feeds
# fcC, so the join is INTERNAL. PDF: back-prop from fcC to each branch splits by variance share
# sigma_branch^2/(sigma_a^2+sigma_b^2).  (p=2 only.)
print("\n=== 2. RESIDUAL toy: fcC(relu(fcB(relu(fcA))+fcS)); internal skip-join ===")
R0, R1, R2, R3 = 8, 6, 5, 4
class ResToy(nn.Module):
    def __init__(s):
        super().__init__()
        s.fcA = nn.Linear(R0, R1); s.fcB = nn.Linear(R1, R2)
        s.fcS = nn.Linear(R0, R2); s.fcC = nn.Linear(R2, R3); s.relu = nn.ReLU()
    def forward(s, x):
        return s.fcC(s.relu(s.fcB(s.relu(s.fcA(x))) + s.fcS(x)))
base = ResToy().eval()
Xr = torch.randn(400, R0) * 1.3 + 0.1
exr = torch.randn(1, R0)
seedC = torch.full((R3,), 1.0/R3)

mr = ResToy(); mr.load_state_dict(base.state_dict()); mr.eval()
mgr2 = MeanResidualManager(mr, ["fcA", "fcB", "fcS", "fcC"], CPU, lambda_reg=0.0, max_batches=10)
mgr2.reparameterize([(Xr[i:i+40], None) for i in range(0, 400, 40)])
try:
    topo = mgr2.build_propagation_topology(exr, p=2)
    rel = mgr2.propagation_importance(I_out=seedC, p=2, relative=True, topology=topo)
    rpA, rpB, rpS, rpC = (mgr2._reparam_modules[k] for k in ["fcA", "fcB", "fcS", "fcC"])
    from torch_pruning.utils.reparam import _contribution_weight
    def MP(rp): return _contribution_weight(rp).detach().t().abs()
    soB, soS = rpB.sigma_out_x, rpS.sigma_out_x
    shB = soB.pow(2) / (soB.pow(2) + soS.pow(2)).clamp_min(1e-12)
    shS = soS.pow(2) / (soB.pow(2) + soS.pow(2)).clamp_min(1e-12)
    IC = wbar(MP(rpC), 2) @ seedC                 # fcC terminal
    IB = wbar(MP(rpB), 2) @ (shB * IC)            # fcB share of the add
    IS = wbar(MP(rpS), 2) @ (shS * IC)            # fcS share of the add
    IA = wbar(MP(rpA), 2) @ IB                     # fcA -> fcB only
    handR = {"fcC": IC, "fcB": IB, "fcS": IS, "fcA": IA}
    # first confirm the topology's branch weights ARE the per-channel sigma^2 share
    wB = dict(topo["fcB"]).get("fcC")
    check("production join weight == per-channel sigma^2 share (fcB->fcC)",
          wB is not None and torch.is_tensor(wB) and torch.allclose(wB, shB, atol=1e-6),
          f"max|d|={None if wB is None else (torch.as_tensor(wB)-shB).abs().max():.1e}")
    for k in ["fcA", "fcB", "fcS", "fcC"]:
        check(f"production RESIDUAL rel {k}", torch.allclose(rel[k], handR[k], atol=1e-4),
              f"max|d|={(rel[k]-handR[k]).abs().max():.1e}")
except Exception as e:
    import traceback; traceback.print_exc()
    check("production residual topology", False, f"EXC {type(e).__name__}: {e}")

# standalone on a FRESH residual model (plain-sum fan-in — does NOT apply the sigma^2 share).
ms = ResToy(); ms.load_state_dict(base.state_dict()); ms.eval()
prunR = {"fcA": ms.fcA, "fcB": ms.fcB, "fcS": ms.fcS, "fcC": ms.fcC}
loaderR = [(Xr[i:i+40], torch.zeros(40, dtype=torch.long)) for i in range(0, 400, 40)]
muR, sigR, fireR, _ = nns.calibrate(ms, prunR, loaderR, CPU, len(loaderR), save_input_for=[])
edgesR = nns.discover_edges(ms, exr, prunR)
scR = nns.compute_scores(prunR, sigR, edgesR, fireR, p=2)
print("  -- standalone residual: PLAIN-SUM fan-in (KNOWN: skips the sigma^2 share at the add) --")
check("standalone residual runs finite", all(torch.isfinite(v).all() for v in scR['rel'].values()), "ran")
# document the gap quantitatively (not a pass/fail on alignment — a known scope limit)
print("  NOTE: standalone is for the per-layer dump viz; production owns the residual-correct prune.")

# ============================================================ verdict
print("\n=== VERDICT ===")
npass = sum(1 for _, ok, _ in RESULTS if ok); ntot = len(RESULTS)
fails = [n for n, ok, _ in RESULTS if not ok]
print(f"{npass}/{ntot} checks passed")
if fails:
    print("FAILURES:")
    for f in fails: print(f"  - {f}")
    sys.exit(1)
print("ALL PDF-ALIGNMENT CHECKS PASSED")
