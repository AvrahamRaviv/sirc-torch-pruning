"""Numerical oracle for the propagation criterion (normalized_nets_pruning.pdf steps 3-10).

Builds a tiny ReLU MLP (no residuals → matches the PDF scalar derivation exactly),
measures sigma_in / sigma_pre / sigma_post EMPIRICALLY, computes the PDF I^l recursion
three ways, and checks which one normnet_standalone.compute_scores actually implements.

PDF (variance, p=2):  M[i,j]=sigma_in_i*|W_ji|,  Wbar=M^2/colsum (colsum_j=sigma_pre_j^2),
  relative      I^l = Wbar @ I^{l+1}
  nonrel(PDF)   I^l = Wbar @ (sigma_POST^2 ⊙ I^{l+1})   <- transfer = POST-activation std
  nonrel(raw)   I^l = Wbar @ (sigma_PRE^2  ⊙ I^{l+1}) = M^2 @ I^{l+1}  (sigma_pre cancels D!)
"""
import sys, os, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("nns", os.path.join(os.path.dirname(__file__), "normnet_standalone.py"))
nns = importlib.util.module_from_spec(spec); spec.loader.exec_module(nns); nns._LOG_FH = None

torch.manual_seed(0)
DIMS = [8, 6, 5, 4, 3]
class MLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.fcs = nn.ModuleList([nn.Linear(DIMS[i], DIMS[i+1]) for i in range(len(DIMS)-1)])
        s.act = nn.ReLU()
    def forward(s, x):
        for k, fc in enumerate(s.fcs):
            x = fc(x)
            if k < len(s.fcs)-1: x = s.act(x)
        return x
m = MLP().eval()

# ---- data + empirical sigma_in / sigma_pre / sigma_post per Linear ----
N = 200
X = torch.randn(N, DIMS[0]) * 1.5 + 0.3
pre, post, inp = {}, {}, {}
def mk_pre(fc):
    def h(mod, i, o): pre.setdefault(fc, []).append(o.detach())
    return h
def mk_in(fc):
    def h(mod, i): inp.setdefault(fc, []).append(i[0].detach())
    return h
hs = []
for fc in m.fcs:
    hs.append(fc.register_forward_hook(mk_pre(fc)))
    hs.append(fc.register_forward_pre_hook(mk_in(fc)))
with torch.no_grad(): m(X)
for h in hs: h.remove()
sig_in  = {fc: torch.cat(inp[fc]).std(0) for fc in m.fcs}
sig_pre = {fc: torch.cat(pre[fc]).std(0) for fc in m.fcs}      # PRE-activation (linear out)
# POST-activation: apply ReLU to pre (except last layer = no activation)
sig_post = {}
for k, fc in enumerate(m.fcs):
    z = torch.cat(pre[fc])
    a = torch.relu(z) if k < len(m.fcs)-1 else z
    sig_post[fc] = a.std(0)

# ---- hand oracle: PDF recursion (p=2), seed = ones/dim at the last layer ----
def oracle(transfer):  # transfer: dict fc->per-output vector (sigma_post^2 or sigma_pre^2 or ones)
    I = {}
    for fc in reversed(list(m.fcs)):
        W = fc.weight.data                      # [out,in]
        M = (W.abs().t() * sig_in[fc][:, None]) # [in,out]
        Mp = M.pow(2); col = Mp.sum(0).clamp_min(1e-12)
        Wbar = Mp / col
        nxt = list(m.fcs)[list(m.fcs).index(fc)+1] if fc is not list(m.fcs)[-1] else None
        I_next = I[nxt] if nxt is not None else torch.full((W.size(0),), 1.0/W.size(0))
        # wait: I_next must be indexed over THIS layer's OUTPUT = next layer's INPUT
        I[fc] = Wbar @ (transfer[fc] * I_next)
    return I
# transfer indexed over each layer's OUTPUT channels:
T_rel  = {fc: torch.ones(fc.weight.size(0)) for fc in m.fcs}
T_post = {fc: sig_post[fc].pow(2) for fc in m.fcs}
T_pre  = {fc: sig_pre[fc].pow(2)  for fc in m.fcs}
# terminal seed convention: I^o normalizes Σ^o → output layer's transfer = ones (matches
# the standalone, which sets σ_post=ones for layers with no in-prunable consumer).
T_post[m.fcs[-1]] = torch.ones(m.fcs[-1].weight.size(0))
T_pre[m.fcs[-1]]  = torch.ones(m.fcs[-1].weight.size(0))

# BUT seed/index bug above: I[nxt] is indexed over nxt's INPUT? In an MLP, layer fc output dim
# == next layer input dim, and the propagated I is per-INPUT-channel of next layer == per-output
# of fc. So I[nxt] (per-input of nxt) aligns with fc's outputs. Good.
rel  = oracle(T_rel)
post = oracle(T_post)
raw  = oracle(T_pre)

# ---- standalone compute_scores on the SAME mlp ----
ex = torch.randn(1, DIMS[0])
loader = [(X[i:i+20], torch.zeros(20, dtype=torch.long)) for i in range(0, N, 20)]
prunable = {f"fcs.{k}": fc for k, fc in enumerate(m.fcs)}
# ignore first+last like the real script? compute_scores scores ALL prunable. Keep all.
mu, sigma, sigma_out, fire, saved = nns.calibrate(m, prunable, loader, torch.device("cpu"), len(loader), save_input_for=[])
edges = nns.discover_edges(m, ex, prunable)
sc = nns.compute_scores(prunable, sigma, sigma_out, mu, edges, fire, p=2)

def drift(d): 
    mn = torch.tensor([v.mean() for v in d.values()]); return (mn.max()/mn.clamp_min(1e-20).min()).item()
print("=== empirical sigma (per layer, mean over channels) ===")
for k, fc in enumerate(m.fcs):
    si = sig_in[fc].mean().item(); sp = sig_pre[fc].mean().item()
    so = sig_post[fc].mean().item(); ratio = (sig_post[fc]/sig_pre[fc]).mean().item()
    print(f"  fc{k}: sigma_in={si:.3f} sigma_pre={sp:.3f} sigma_post={so:.3f}  (post/pre={ratio:.3f})")
print(f"\nstandalone sigma_out vs sig_pre vs sig_post (fc0): out={sigma_out[m.fcs[0]].mean():.3f} pre={sig_pre[m.fcs[0]].mean():.3f} post={sig_post[m.fcs[0]].mean():.3f}")

print("\n=== match standalone nonrel against the 3 oracles (fc0 per-input score) ===")
sa = sc['nonrel'][m.fcs[0]]
for name, o in [("rel",rel),("nonrel_POST(PDF)",post),("nonrel_raw(=sigma_pre)",raw)]:
    diff = (sa - o[m.fcs[0]]).abs().max().item()
    print(f"  standalone nonrel vs {name:22s}: max|diff|={diff:.3e} {'<-- MATCH' if diff<1e-4 else ''}")
print(f"\n=== drift (max/min per-layer mean) ===")
print(f"  oracle rel        = {drift(rel):.3e}")
print(f"  oracle nonrel POST = {drift(post):.3e}   (PDF-correct transfer)")
print(f"  oracle nonrel raw  = {drift(raw):.3e}   (sigma_pre cancels D)")
print(f"  standalone rel     = {drift(sc['rel']):.3e}")
print(f"  standalone nonrel  = {drift(sc['nonrel']):.3e}")
