"""
Residual-MLP-on-MNIST check of the REAL propagation pipeline (reparam.py), 4 configs:
  base   nonrel, no corrections
  cov    + --prop_measured_var  (layer denom = measured Var(Z_j))
  sigmac + --skip_sigma_c        (join denom = PDF σ_c^p/(σ_a^p+σ_b^p))
  both

Two questions:
  1. MATH — does propagation_importance rank channels like the BRUTE-FORCE drop-one contribution
     to final-output variance? (center each layer-input channel, measure Δ Σ_classes Var(logit);
     compare to the criterion via Spearman corr.) Residual blocks give real joins so σ_c bites.
  2. COMPOUND — per-layer score magnitude across DEPTH. nonrel is documented to compound (∏σ_post);
     check whether base explodes and whether cov/σ_c (measured node var) tame or worsen it.

Run: python benchmarks/vbp/mlp_prop_check.py            (trains ~2 ep, then the 4-config report)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class ResMLP(nn.Module):
    """784→W, then L residual blocks  x = x + fc2(relu(fc1(x)))  (constant width → real adds),
    head W→10. The adds are the residual joins build_propagation_topology must weight."""
    def __init__(self, width=128, depth=4):
        super().__init__()
        self.embed = nn.Linear(784, width)
        self.fc1 = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.fc2 = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.head = nn.Linear(width, 10)
        self.depth = depth

    def forward(self, x):
        x = F.relu(self.embed(x.view(x.size(0), -1)))
        for i in range(self.depth):
            x = x + self.fc2[i](F.relu(self.fc1[i](x)))   # residual join
        return self.head(x)


def loaders(batch=256):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    te = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0),
            torch.utils.data.DataLoader(te, batch_size=1024, shuffle=False, num_workers=0))


def train(model, tr, te, dev, epochs):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            opt.zero_grad(); F.cross_entropy(model(x.to(dev)), y.to(dev)).backward(); opt.step()
        model.eval(); n = c = 0
        with torch.no_grad():
            for x, y in te:
                c += (model(x.to(dev)).argmax(1) == y.to(dev)).sum().item(); n += y.numel()
        print(f"  ep{ep} test_acc {c/n:.4f}")


def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm() + 1e-12)).item()


@torch.no_grad()
def brute_importance(model, names, Xcache):
    """True importance of each scored layer's INPUT channels = drop-one Δ Σ_classes Var(logit).
    Center channel c at the layer's input over the eval batch, recompute logits, measure ΔVar.
    Uses a forward hook to overwrite the input of the target layer. Returns {name: tensor}."""
    mods = dict(model.named_modules())
    base_logits = _forward_collect(model, Xcache, None, None)
    base_var = base_logits.var(0, unbiased=False).sum().item()
    out = {}
    for nm in names:
        W_in = mods[nm].in_features
        idxs = list(range(0, W_in, max(1, W_in // 12)))[:12]
        imp = torch.zeros(W_in)
        for c in idxs:
            lv = _forward_collect(model, Xcache, mods[nm], c)
            imp[c] = base_var - lv.var(0, unbiased=False).sum().item()
        out[nm] = (imp, idxs)
    return out


@torch.no_grad()
def _forward_collect(model, X, target, ch):
    """Forward, optionally centering input channel `ch` of `target` module via a pre-hook."""
    h = None
    if target is not None:
        mean_c = None
        def pre(mod, inp):
            x = inp[0].clone(); x[:, ch] = x[:, ch].mean(); return (x,)
        h = target.register_forward_pre_hook(pre)
    logits = model(X)
    if h: h.remove()
    return logits


def main():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from vbp_common import build_whole_net_reparam_layers
    from torch_pruning.utils.reparam import MeanResidualManager

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tr, te = loaders()
    model = ResMLP(128, 4).to(dev)
    ckpt = os.path.join(os.path.dirname(__file__), "resmlp_mnist.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=dev)); print("loaded", ckpt)
    else:
        print("training ResMLP..."); train(model, tr, te, dev, 2); torch.save(model.state_dict(), ckpt)

    # one eval batch for brute-force (cpu, big enough for stable variance)
    Xb = torch.cat([x for x, _ in te][:4]).to(dev)[:3000]

    ex = torch.zeros(1, 1, 28, 28, device=dev)
    configs = {"base": (False, False), "cov": (True, False),
               "sigmac": (False, True), "both": (True, True)}

    for cname, (mv, sc) in configs.items():
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        names = build_whole_net_reparam_layers(model, exclude_classifier=True, exclude_stem=False)
        mgr = MeanResidualManager(model, names, dev, max_batches=20)
        mgr.reparameterize(tr)
        topo = mgr.build_propagation_topology(ex, p=2, use_measured_sigma_c=sc)
        I = mgr.propagation_importance(p=2, topology=topo, relative=False, use_measured_var=mv)
        mgr.merge_back()

        # 1. MATH: Spearman(propagation-I, brute drop-one) per scored layer
        brute = brute_importance(model, [n for n in names if n in I], Xb)
        cors = []
        for nm, (bimp, idxs) in brute.items():
            pi = I[nm][idxs].cpu()
            cors.append(spearman(pi, bimp[idxs]))
        mcor = sum(cors) / len(cors)

        # 2. COMPOUND: per-layer mean score across depth (forward order)
        scale = [(nm, float(I[nm].mean())) for nm in names if nm in I]
        vals = [v for _, v in scale]
        ratio = max(vals) / (min(v for v in vals if v > 0) + 1e-12)
        print(f"\n=== {cname}  (measured_var={mv}, skip_sigma_c={sc}) ===")
        print(f"  MATH: mean Spearman(prop-I, brute drop-one) = {mcor:+.3f}  "
              f"(per-layer {[round(c,2) for c in cors]})")
        print(f"  COMPOUND: per-layer mean-I across depth (max/min ratio = {ratio:.2e}):")
        print(f"    {[round(v,3) if v>=1e-3 else f'{v:.1e}' for _, v in scale]}")

    # --- REFERENCE: one-hop criteria vs the SAME brute ground truth (config-independent) ---
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    mods = dict(model.named_modules())
    sig = {}
    hs = []
    acc = {}
    def mk(nm):
        def pre(mod, inp):
            a = inp[0].detach(); acc.setdefault(nm, []).append(a.var(0))
        return pre
    for nm in names:
        if nm in I:
            hs.append(mods[nm].register_forward_pre_hook(mk(nm)))
    with torch.no_grad():
        for x, _ in tr:
            model(x.to(dev))
            if len(next(iter(acc.values()))) >= 20:
                break
    for h in hs:
        h.remove()
    sig = {nm: torch.stack(v).mean(0) for nm, v in acc.items()}   # per-input-channel var
    brute = brute_importance(model, [n for n in names if n in I], Xb)
    nci_c, mag_c = [], []
    for nm, (bimp, idxs) in brute.items():
        W = mods[nm].weight.data
        mag = W.pow(2).sum(0).cpu()                               # ‖W[:,c]‖²  (own-layer magnitude)
        nci = (sig[nm].cpu() * mag)                               # σ_c²·‖W[:,c]‖²  one-hop NCI
        nci_c.append(spearman(nci[idxs], bimp[idxs]))
        mag_c.append(spearman(mag[idxs], bimp[idxs]))
    print(f"\n=== REFERENCE one-hop vs brute drop-one (same ground truth) ===")
    print(f"  magnitude ‖W[:,c]‖²  mean Spearman = {sum(mag_c)/len(mag_c):+.3f}")
    print(f"  NCI σ_c²‖W[:,c]‖²    mean Spearman = {sum(nci_c)/len(nci_c):+.3f}")
    print(f"  (propagation configs above were ~0.69-0.70)")


if __name__ == "__main__":
    main()
