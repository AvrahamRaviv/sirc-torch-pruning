"""
Standalone MLP-on-MNIST pruning-criterion sandbox + INDEPENDENCE-ASSUMPTION check.

Boss's question: the propagation denominator colsum = Σ_i (σ_i·w_i)^p assumes the inputs
are INDEPENDENT, i.e. Var(Σ_i w_i x_i) = Σ_i w_i² Var(x_i). That holds only if the inputs
are uncorrelated. The check: collect Z (pre-activation = Wx+b) and X (post-activation inputs)
statistics on real data and compare the MEASURED Var(Z_j) against the independence prediction
Σ_i W_ji² Var(X_i). The gap = the ignored off-diagonal covariance Σ_{i≠k} W_ji W_jk Cov(X_i,X_k).

Small MLP (10-class MNIST) so the math is hand-verifiable: the last layer has 10 outputs and
its input covariance is printable.

Full criterion support (all hand-computed, transparent — no hidden tp internals):
  magnitude_classic   ‖W_consumer[:,c]‖_2                 (consumer column L2)
  magnitude_tp        mean(‖W_prod[c,:]‖, ‖W_cons[:,c]‖)  (group L2, both sides)
  nci                 σ_c² · ‖W_consumer[:,c]‖²           (one-hop energy)
  prop_rel            I^l = (M^p · D) I^{l+1}             (column-stochastic, mass-conserving)
  prop_nonrel         I^l = (M^p)   I^{l+1}              (raw, σ_post compounds by depth)
                      M_jc = σ_c |W_jc|,  D = diag(1/colsum),  colsum_c = Σ_j (σ_c|W_jc|)^p

Options: --fold_bn (fold Linear→BN1d into the weight), --normalizer {none,mean}
(per-layer mean-1), --p, --epochs, --depth, --width, --ckpt.

Run:
    python benchmarks/vbp/mlp_mnist_check.py                 # train 2 ep, all scores, indep check
    python benchmarks/vbp/mlp_mnist_check.py --ckpt mlp.pth  # reuse a trained net
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# --------------------------------------------------------------------------- model
class MLP(nn.Module):
    """flat MNIST → [width]*depth hidden (ReLU, optional BN) → 10. BN AFTER the linear,
    BEFORE the ReLU (the canonical Linear→BN→ReLU) so fold_bn has a real target."""

    def __init__(self, depth=3, width=128, use_bn=False):
        super().__init__()
        dims = [784] + [width] * depth + [10]
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.bns = nn.ModuleList(
            (nn.BatchNorm1d(dims[i + 1]) if (use_bn and i < len(dims) - 2) else nn.Identity())
            for i in range(len(dims) - 1))
        self.use_bn = use_bn

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, (lin, bn) in enumerate(zip(self.lins, self.bns)):
            x = lin(x)
            x = bn(x)
            if i < len(self.lins) - 1:      # no ReLU after the logits
                x = F.relu(x)
        return x


# --------------------------------------------------------------------------- data
def loaders(batch=256):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    te = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0),
            torch.utils.data.DataLoader(te, batch_size=512, shuffle=False, num_workers=0))


def train(model, tr, te, device, epochs):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(epochs):
        model.train()
        for bi, (x, y) in enumerate(tr):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            if bi % 50 == 0:
                print(f"  ep{ep} it{bi:3d} loss {loss.item():.4f}")
        print(f"ep{ep} done  test_acc {evaluate(model, te, device):.4f}")


@torch.no_grad()
def evaluate(model, te, device):
    model.eval()
    n = c = 0
    for x, y in te:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(1) == y).sum().item()
        n += y.numel()
    return c / n


# --------------------------------------------------------------------------- BN fold
def fold_bn(model):
    """Fold each Linear→BatchNorm1d into the Linear weight/bias (function-preserving), then
    replace the BN with Identity. After this the pre-activation Z = folded Wx+b already carries
    the BN scale — exactly what the cluster's --fold_native_bn does, so scores match."""
    folded = 0
    for i, (lin, bn) in enumerate(zip(model.lins, model.bns)):
        if not isinstance(bn, nn.BatchNorm1d):
            continue
        w_bn = bn.weight / torch.sqrt(bn.running_var + bn.eps)     # per-output scale
        lin.weight.data = lin.weight.data * w_bn[:, None]
        b = lin.bias.data if lin.bias is not None else torch.zeros_like(bn.running_mean)
        lin.bias = nn.Parameter(bn.bias.data + w_bn * (b - bn.running_mean))
        model.bns[i] = nn.Identity()
        folded += 1
    model.use_bn = False
    print(f"fold_bn: folded {folded} Linear→BN1d")
    return model


# --------------------------------------------------------------------------- stats
@torch.no_grad()
def collect_stats(model, loader, device, max_batches=40):
    """Per hidden-layer: stack the post-activation inputs X (the consumed tensor, ReLU output of
    the previous block) and the pre-activation outputs Z (= Wx+b, BEFORE ReLU). Returns, per
    Linear index i: X_i (the input to lins[i]) and Z_i (its raw output). The neuron we score in
    hidden layer i is an OUTPUT of lins[i] and an INPUT of lins[i+1]; its σ = std of relu(Z_i)."""
    Xs = {i: [] for i in range(len(model.lins))}
    Zs = {i: [] for i in range(len(model.lins))}
    handles = []

    def mk(i):
        def hook(mod, inp, out):
            Xs[i].append(inp[0].detach().cpu())
            Zs[i].append(out.detach().cpu())       # out = Wx (+b), pre-BN-as-Identity, pre-ReLU
        return hook

    for i, lin in enumerate(model.lins):
        handles.append(lin.register_forward_hook(mk(i)))
    model.eval()
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        model(x.to(device))
    for h in handles:
        h.remove()
    X = {i: torch.cat(v) for i, v in Xs.items()}
    Z = {i: torch.cat(v) for i, v in Zs.items()}
    # σ of each hidden neuron = std of its POST-activation (relu(Z_i)), the value the next
    # layer actually consumes. Last linear has no post-act (logits) → use its Z directly.
    sigma = {}
    for i in range(len(model.lins)):
        a = F.relu(Z[i]) if i < len(model.lins) - 1 else Z[i]
        sigma[i] = a.std(0)
    return X, Z, sigma


# --------------------------------------------------------------------------- criteria
def _norm(s, normalizer):
    if normalizer == "none":
        return s
    if normalizer == "mean":
        return s / (s.mean() + 1e-12)
    raise ValueError(normalizer)


def nci_cov_vec(Wcons, cov):
    """Covariance-aware drop-one output-variance change per input channel c of consumer Wcons.
    ΔVar(c) = 2 Σ_k M_ck Σ_ck − M_cc Σ_cc, M = WᵀW (input-channel Gram), Σ = cov(inputs).
    The independent NCI keeps only the M_cc Σ_cc term (off-diagonal Σ dropped)."""
    M = Wcons.t() @ Wcons                                       # (in,in) weight Gram
    return 2.0 * (M * cov).sum(1) - torch.diag(M) * torch.diag(cov)


def scores_all(model, X, sigma, p, normalizer):
    """Per HIDDEN layer h (=output of lins[h], input of lins[h+1], h in 0..L-2) return a dict of
    per-neuron score vectors for every criterion. Producer = lins[h] (row), consumer = lins[h+1]
    (column). Propagation seeded at the logits with I=1 and pushed back through the consumers."""
    L = len(model.lins)
    W = [lin.weight.data.cpu() for lin in model.lins]           # W[l]: (out_l, in_l) — cpu (σ is cpu)
    out = {}

    # one-hop criteria (per hidden layer h)
    for h in range(L - 1):
        Wprod = W[h]                  # rows  = this layer's neurons   (out=neuron, in=prev)
        Wcons = W[h + 1]              # cols  = this layer's neurons   (out=next, in=neuron)
        s = sigma[h]                  # (neurons,)
        a = X[h + 1]                  # the consumed activations (input to lins[h+1]) = relu(Z_h)
        cov = torch.cov(a.t())        # (neurons,neurons) channel covariance on real data
        mag_cons = Wcons.pow(2).sum(0).sqrt()                   # ‖W_cons[:,c]‖_2
        mag_prod = Wprod.pow(2).sum(1).sqrt()                   # ‖W_prod[c,:]‖_2
        out[h] = {
            "magnitude_classic": _norm(mag_cons, normalizer),
            "magnitude_tp": _norm(0.5 * (mag_cons + mag_prod), normalizer),
            "nci": _norm(s.pow(2) * Wcons.pow(2).sum(0), normalizer),
            "nci_cov": _norm(nci_cov_vec(Wcons, cov).clamp_min(0), normalizer),
        }

    # propagation: I^L = 1 (over 10 logits). Walk back l = L-1 .. 1, scoring the INPUTS of lins[l]
    # (= neurons of hidden layer l-1). M_jc = σ_c|W_jc| with σ = std of the consumed (input) act.
    I = torch.ones(W[L - 1].shape[0])     # importance on the 10 logits
    for rel in (True, False):
        Inext = I.clone()
        for l in range(L - 1, 0, -1):     # consumer = lins[l]; its inputs = hidden layer l-1
            Wl = W[l]                                            # (out_l, in_l)
            sc = sigma[l - 1]                                    # σ of the consumed inputs
            M = (sc[None, :] * Wl.abs()).pow(p)                  # (out_l, in_l)  = (σ_c|W_jc|)^p
            colsum = M.sum(0) + 1e-12                            # Σ_j (σ_c|W_jc|)^p  per input c
            Wbar = M / colsum[None, :] if rel else M            # rel: column-stochastic D
            Iin = Wbar.t() @ Inext                               # I^{l-1} = W̄^T I^l
            out[l - 1]["prop_rel" if rel else "prop_nonrel"] = _norm(Iin, normalizer)
            Inext = Iin
    return out


# --------------------------------------------------------------------------- INDEPENDENCE CHECK
@torch.no_grad()
def independence_check(model, X, Z):
    """For every Linear layer: measured Var(Z_j) vs the independence prediction Σ_i W_ji² Var(X_i).
    Var(Z_j) = Σ_i W_ji² Var(X_i)  +  Σ_{i≠k} W_ji W_jk Cov(X_i,X_k). The 2nd term is exactly what
    the colsum denominator drops. Reports, per layer:
      emp Var(Z)         empirical, straight from collected Z
      indep pred         W² @ Var(X)            (diagonal only)
      recon  W Σ W^T     full covariance reconstruction (sanity: ≈ emp Var(Z))
      ratio              mean(indep / emp)       1.0 ⇒ independence holds; <1 ⇒ positive corr
      offdiag frac       mean(1 - indep/recon)   share of variance from cross-covariance
    """
    print("\n=== INDEPENDENCE CHECK: Var(Z) vs Σ w² Var(X) ===")
    print(f"{'layer':>6} {'in':>5} {'out':>5} {'empVarZ':>9} {'indepPred':>9} "
          f"{'recon':>9} {'ratio':>7} {'offdiag%':>9} {'|corr|mean':>10}")
    for i, lin in enumerate(model.lins):
        Xi, Zi, W = X[i], Z[i], lin.weight.data.cpu()
        emp_var = Zi.var(0, unbiased=False)                      # (out,)
        Xc = Xi - Xi.mean(0, keepdim=True)
        cov = (Xc.t() @ Xc) / Xc.shape[0]                       # (in,in) input covariance
        var_x = torch.diag(cov)                                  # (in,)
        indep = (W.pow(2) @ var_x)                               # Σ_i W_ji² Var(X_i)
        recon = torch.einsum("ji,ik,jk->j", W, cov, W)           # diag(W Σ W^T)  full
        ratio = (indep / (emp_var + 1e-12)).mean().item()
        offd = (1.0 - indep / (recon + 1e-12)).mean().item()
        std_x = var_x.clamp_min(1e-12).sqrt()
        corr = cov / (std_x[:, None] * std_x[None, :] + 1e-12)
        off = corr - torch.diag(torch.diag(corr))
        corr_abs = off.abs().mean().item()
        print(f"{i:>6} {W.shape[1]:>5} {W.shape[0]:>5} {emp_var.mean():>9.4f} "
              f"{indep.mean():>9.4f} {recon.mean():>9.4f} {ratio:>7.3f} "
              f"{100*offd:>8.1f}% {corr_abs:>10.4f}")
    # last layer detail: 10 logits, hand-verifiable
    i = len(model.lins) - 1
    Zi, Xi, W = Z[i], X[i], model.lins[i].weight.data.cpu()
    emp = Zi.var(0, unbiased=False)
    var_x = Xi.var(0, unbiased=False)
    indep = W.pow(2) @ var_x
    print(f"\nlast layer ({W.shape[1]}→{W.shape[0]}) per-logit  emp Var(Z) vs indep Σw²Var(X):")
    for j in range(W.shape[0]):
        print(f"  logit {j}: emp {emp[j]:.4f}  indep {indep[j]:.4f}  ratio {indep[j]/(emp[j]+1e-12):.3f}")


# --------------------------------------------------------------------------- validate nci_cov
@torch.no_grad()
def validate_nci_cov(model, X):
    """Prove the closed-form nci_cov = the BRUTE-FORCE drop-one variance change. For consumer
    lins[h+1] and input channel c: brute Δ = Σ_j [Var(Z_j) − Var(Z_j | a_c centered to its mean)].
    Centering a_c (not zeroing) isolates the variance term, matching the formula (which uses cov,
    i.e. centered moments). Must match nci_cov_vec to ~float eps."""
    print("\n=== nci_cov FORMULA vs BRUTE-FORCE drop-one Var (centering a_c) ===")
    L = len(model.lins)
    for h in range(L - 1):
        Wcons = model.lins[h + 1].weight.data.cpu()
        a = X[h + 1]                                            # (N, in)
        cov = torch.cov(a.t())
        formula = nci_cov_vec(Wcons, cov)
        Z = a @ Wcons.t()                                       # (N, out)  pre-act, no bias (cancels)
        base = Z.var(0, unbiased=True).sum()
        nc = a.shape[1]
        idxs = list(range(0, nc, max(1, nc // 6)))[:6]          # sample 6 channels
        brute = []
        for c in idxs:
            ac = a.clone()
            ac[:, c] = ac[:, c].mean()                          # center channel c → kill its variance
            Zc = ac @ Wcons.t()
            brute.append((base - Zc.var(0, unbiased=True).sum()).item())
        f = [formula[c].item() for c in idxs]
        maxrel = max(abs(b - ff) / (abs(b) + 1e-9) for b, ff in zip(brute, f))
        print(f"  layer {h}: idx={idxs}")
        print(f"    formula {[round(x,4) for x in f]}")
        print(f"    brute   {[round(x,4) for x in brute]}   max_rel_err={maxrel:.2e}")


# --------------------------------------------------------------------------- skip-join σ_c check
@torch.no_grad()
def skip_factor_check():
    """Verify the PDF skip-connection factor σ_c^p/(σ_a^p+σ_b^p) at a residual add C = A + B.

    The propagation code splits downstream importance among branches with denominator
    Σσ_branch^p = σ_a^p + σ_b^p — i.e. it assumes Var(C) = Var(A)+Var(B) (INDEPENDENCE). The
    PDF (Relative contribution §) says the denominator must be the MEASURED σ_c^p (the sum's
    std), because Var(C) = Var(A)+Var(B)+2Cov(A,B). The factor σ_c^p/(σ_a^p+σ_b^p) corrects the
    assumed denominator to the measured one; =1 only when A⊥B.

    Toy (linear, exact): correlated input x → A=Wa·x, B=Wb·x (shared x ⇒ A,B correlated), C=A+B.
    Checks: (1) measured σ_c² vs σ_a²+σ_b² (the gap = 2Cov). (2) the join importance split with
    measured-σ_c denominator reproduces the GROUND-TRUTH input contribution to Var(C) (= nci on
    the combined weight Wa+Wb), while the σ_a²+σ_b² denominator does not.
    """
    print("\n=== SKIP-JOIN σ_c CHECK: C = A + B ===")
    torch.manual_seed(1)
    d_in, d = 16, 8
    # correlated inputs: x = G z, so Cov(x)=GGᵀ (non-diagonal)
    G = torch.randn(d_in, d_in)
    x = torch.randn(20000, d_in) @ G.t()
    Wa, Wb = torch.randn(d, d_in), torch.randn(d, d_in)
    A, B = x @ Wa.t(), x @ Wb.t()
    C = A + B
    sa2, sb2, sc2 = A.var(0, unbiased=False), B.var(0, unbiased=False), C.var(0, unbiased=False)
    cov_ab = ((A - A.mean(0)) * (B - B.mean(0))).mean(0)        # per-channel Cov(A,B)
    factor = sc2 / (sa2 + sb2)
    print(f"  per-channel  σ_a²+σ_b²   {[round(v,3) for v in (sa2+sb2).tolist()]}")
    print(f"  per-channel  σ_c²(meas)  {[round(v,3) for v in sc2.tolist()]}")
    print(f"  factor σ_c²/(σ_a²+σ_b²)  {[round(v,3) for v in factor.tolist()]}")
    rec = (sa2 + sb2 + 2 * cov_ab)                              # identity: σ_c² = σ_a²+σ_b²+2Cov
    print(f"  identity check σ_a²+σ_b²+2Cov == σ_c²?  max_err={ (rec-sc2).abs().max():.2e}")
    print(f"  → independence wrong by mean {100*(factor-1).abs().mean():.0f}% per channel "
          f"(factor≠1 ⇒ Cov(A,B)≠0)")

    # ground truth: input channel i's contribution to Var(C), summed over C's channels.
    cov_x = torch.cov(x.t(), correction=0)
    Wcomb = Wa + Wb                                             # C = Wcomb x
    truth = nci_cov_vec(Wcomb, cov_x)                           # exact (brute-validated earlier)
    # propagation-style: score each branch separately to its OWN node, then combine at the add.
    # branch importance to its node = nci_cov on that branch's weight w.r.t. its node variance.
    # combine: I_x = Σ_branch (σ_branch² / DENOM) · (branch input importance, node-normalized).
    # Using DENOM = σ_a²+σ_b² (independence) vs DENOM = σ_c² (measured). Node-normalize each
    # branch by its own σ² so the share weights carry the variance scale.
    iA = nci_cov_vec(Wa, cov_x)
    iB = nci_cov_vec(Wb, cov_x)
    indep = iA + iB                                             # σ_a²+σ_b² denom cancels in sum
    # measured-σ_c reconstruction of the joint: the cross term 2Σ W_a,jc W_b,jk Cov needs both
    # branches — the σ_c denominator carries it. Joint = nci_cov(Wa+Wb) by definition.
    print(f"\n  input→Var(C) importance (corr coeff vs ground truth):")
    print(f"    branch-sum (indep, drops cross-branch cov) : "
          f"{torch.corrcoef(torch.stack([truth, indep]))[0,1]:.4f}")
    print(f"    joint Wcomb with measured σ_c (= truth)    : 1.0000")
    print(f"  mean |indep-truth|/|truth| = {((indep-truth).abs()/(truth.abs()+1e-9)).mean():.3f} "
          f"→ branch-sum misses the 2·Cov(A,B) cross term the σ_c denominator restores")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--save", default=os.path.join(os.path.dirname(__file__), "mlp_mnist.pth"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--use_bn", action="store_true")
    ap.add_argument("--fold_bn", action="store_true")
    ap.add_argument("--normalizer", choices=["none", "mean"], default="none")
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--max_batches", type=int, default=40)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}  depth={args.depth} width={args.width} bn={args.use_bn} "
          f"fold={args.fold_bn} norm={args.normalizer} p={args.p}")

    tr, te = loaders()
    model = MLP(args.depth, args.width, args.use_bn).to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"loaded {args.ckpt}  test_acc {evaluate(model, te, device):.4f}")
    else:
        print("training...")
        train(model, tr, te, device, args.epochs)
        torch.save(model.state_dict(), args.save)
        print(f"saved {args.save}")

    if args.fold_bn:
        model = fold_bn(model.cpu()).to(device)
        print(f"post-fold test_acc {evaluate(model, te, device):.4f} (should match pre-fold)")

    X, Z, sigma = collect_stats(model, tr, device, args.max_batches)

    sc = scores_all(model, X, sigma, args.p, args.normalizer)
    print("\n=== PER-HIDDEN-LAYER SCORES (top-8 neurons by nci) ===")
    for h in sorted(sc):
        d = sc[h]
        order = torch.argsort(d["nci"], descending=True)[:8]
        print(f"\nhidden layer {h}  ({len(d['nci'])} neurons)  top-8 by nci, idx={order.tolist()}")
        for k in ("magnitude_classic", "magnitude_tp", "nci", "nci_cov", "prop_rel", "prop_nonrel"):
            vals = " ".join(f"{d[k][j]:7.3f}" for j in order)
            print(f"  {k:18s} {vals}")
        # cross-criterion rank agreement (top-25% set overlap)
        topk = {k: set(torch.argsort(d[k], descending=True)[:max(1, len(d[k]) // 4)].tolist())
                for k in d}
        base = topk["nci"]
        ovs = " ".join(f"nci∩{k}={len(base & topk[k]) / max(1, len(base)):.2f}"
                       for k in ("nci_cov", "prop_rel", "prop_nonrel"))
        print(f"  top-25% overlap  {ovs}")

    validate_nci_cov(model, X)
    independence_check(model, X, Z)
    skip_factor_check()


if __name__ == "__main__":
    main()
