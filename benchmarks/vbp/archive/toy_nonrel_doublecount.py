#!/usr/bin/env python
"""Toy check: does our non-relative propagation double-count sigma?

Data-independent algebra on a random MLP. No ImageNet needed — the question is
purely about which sigma factors appear in the matrix chain.

PDF (normalized_nets_pruning.pdf) forms:
  std/var propagation (step 7-8):
      I^l = Σ^l W^l D^l Σ^{l+1} W^{l+1} D^{l+1} … Σ^o I^o
      with W' = ΣW (sigma folded into W' ONCE), W̄ = W'^p · D, D = 1/colsum(W'^p).
      => I^l = W̄^l … W̄^L I^o   (NO separate per-hop sigma transfer; each σ appears once)
  PDF page-3: "Variance propagation yields the SAME relative criterion for p=2."

Our code (reparam.py) builds M = ṽ = σ·W = W'  (correct), W̄ = M^p/colsum (correct),
then for non-relative ALSO does  I_next ← σ_post^p ⊙ I_next  before the matmul.
σ_post[l] = σ_in[l+1], which is ALREADY inside M^{l+1}=σ_in[l+1]·W  → applied twice.

This script builds three chains and compares:
  rel        : ∏ W̄                       (column-stochastic, no transfer)
  pdf        : literal Σ W D … Σ W D form (raw W + separate σ diagonals, σ once)
  cur_nonrel : ∏ W̄ with σ_post^p transfer (what our code does today)
"""
import torch

torch.manual_seed(0)

# random MLP: widths in -> ... -> out (out = "classes")
widths = [10, 8, 7, 6, 5]          # layer l input dims; last = output/classes
L = len(widths) - 1                 # number of weight layers
W = [torch.randn(widths[i + 1], widths[i]).abs() for i in range(L)]   # [out,in], >0
sigma = [torch.rand(widths[i]) * 1.5 + 0.2 for i in range(L)]         # per-INPUT std of layer i
# post-activation output std of layer l == input std of layer l+1 (== sigma[l+1]).
# terminal layer's "next input std" = the output-side std; pick a random one.
sigma_out_last = torch.rand(widths[L]) * 1.5 + 0.2

p = 2
eps = 1e-12


def sigma_post(l):
    """post-act output std of weight-layer l = input std of layer l+1 (sigma[l+1]),
    or the terminal output std for the last layer."""
    return sigma[l + 1] if l + 1 < L else sigma_out_last


def Wbar(l):
    """column-stochastic W̄ = M^p / colsum, M[in,out] = sigma_in[in]*W[out,in]."""
    M = (sigma[l][:, None] * W[l].t()).abs()      # [in, out]
    Mp = M.pow(p)
    return Mp / Mp.sum(dim=0, keepdim=True).clamp(min=eps)


def seed():
    # output seed: uniform over classes, weighted by output std^p (Σ^o I^o, the step-8 seed)
    n = widths[L]
    return (sigma_out_last.pow(p)) * torch.full((n,), 1.0 / n)


# ---- rel: pure ∏ W̄, no transfer -------------------------------------------
def chain_rel():
    I = seed()
    out = {}
    for l in reversed(range(L)):
        I = Wbar(l) @ I
        out[l] = I
    return out


# ---- cur_nonrel: ∏ W̄ with the extra σ_post^p ⊙ I_next (our code today) -----
def chain_cur_nonrel():
    I = seed()
    out = {}
    for l in reversed(range(L)):
        I = sigma_post(l).pow(p) * I        # <-- the extra transfer (suspected double-count)
        I = Wbar(l) @ I
        out[l] = I
    return out


# ---- pdf: literal Σ^l W^l D^l Σ^{l+1} … with RAW W and separate σ, σ once ----
# variance form (p=2): per layer operator on input-importance vector.
# M'[in,out] = (sigma_in[in]*W[out,in])^2 = W''  ; D = 1/colsum(M') ; W̄_pdf = M'·D.
# This is identical to Wbar(l) for p=2 BY CONSTRUCTION (sigma folded once, no transfer).
def chain_pdf():
    I = seed()
    out = {}
    for l in reversed(range(L)):
        M = (sigma[l][:, None] * W[l].t()).abs()
        Mp = M.pow(p)
        Wb = Mp / Mp.sum(dim=0, keepdim=True).clamp(min=eps)   # σ appears once, no transfer
        I = Wb @ I
        out[l] = I
    return out


def span(v):
    v = v[v > 0]
    return (v.max() / v.min()).item() if v.numel() else float("nan")


rel = chain_rel()
cur = chain_cur_nonrel()
pdf = chain_pdf()

print(f"p={p}, {L} weight layers, widths={widths}\n")
print(f"{'layer':>5} {'rel span':>12} {'pdf span':>12} {'cur span':>14}   cur/rel ratio range")
for l in range(L):
    r = rel[l]; c = cur[l]; q = pdf[l]
    ratio = (c / r.clamp(min=eps))
    print(f"{l:>5} {span(r):>12.3f} {span(q):>12.3f} {span(c):>14.3e}   "
          f"[{ratio.min():.2e}, {ratio.max():.2e}]")

# the two algebraic claims:
rel_eq_pdf = max((rel[l] - pdf[l]).abs().max().item() for l in range(L))
print(f"\n max|rel - pdf|                     = {rel_eq_pdf:.2e}   "
      f"(PDF: variance prop == relative for p=2 → expect ~0)")

# cur should equal rel times an accumulating ∏ σ_post^p factor (the double count).
# predicted extra factor at layer l = ∏_{k=l}^{L-1} σ_post(k)^p  applied through the chain.
print("\n cur_nonrel adds an extra σ_post^p per hop that rel/pdf do NOT have.")
print(" If cur != rel while rel == pdf, the extra σ_post transfer is the double-count.")
