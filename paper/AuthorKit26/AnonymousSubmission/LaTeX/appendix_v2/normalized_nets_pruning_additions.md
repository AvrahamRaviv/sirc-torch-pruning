# Normalized-Nets Pruning — Report Additions

Companion to `normalized_nets_pruning.pdf`. Four parts: (1) errata, (2) implementation,
(3) an optimization-geometry view, (4) where the transform lives in code.

---

## 1. Errata / issues in the current report

**Real errors**

- **Matrix `D` normalization index.** In the relative-contribution matrix form,
  `D = Diag(…, 1/Σ_j w'_ij, …)` sums over the *output* index `j`. To match the scalar
  definition (1) `σ_i²w_ij² / Σ_î σ_î²w_îj²` and "normalize each column", it must sum over
  the *input* index: **`D = Diag(…, 1/Σ_i w'_ij, …)`**. The code is correct
  (`Mp.sum(dim=0)` = sum over input channels).

- **L2/L1 equivalence bullet.** "Regularizing the L2 norm of rows of `W'` ≡ L1 norm of
  rows of `W''` for p=2" holds only as the **squared** L2 norm
  (`‖σW row‖² = Σσ²w² = L1 of W''`). State it as L2², or drop the wording.

- **Subscript typo, formula (2).** Denominator `Σ_î σ_i w_ij` should read
  `Σ_î σ_î w_îj` (sum over `î`, both factors indexed by `î`).

**Approximations stated as facts (label as modeling assumptions, not derivations)**

- **Input independence** (step 2, "variance is additive") ignores cross-channel
  correlation.
- **Linearized activation** (steps 3–4): the activation is collapsed to a single scalar
  std-gain `f = σ^{l+1}/σ^l`. The whole `D`-chain rides on this; it is an average-case
  linearization, exact only for homogeneous (e.g. ReLU on zero-mean) activations.
- **"77% pre-act = 77% post-act"** is exact only because `f_j` is common to all inputs of
  output `j` and cancels in the ratio — true under the linearized-`f` model, not per-sample.

**Missing discussion**

- The report assumes a clean conv→act net. It never addresses that a **BN net already
  normalizes the pre-activation** — yet the conv consumes the *post*-activation signal
  (mean `relu(β)`, std `≈0.58γ`), which BN never touches. This is exactly why the transform
  still earns its keep on ResNet (see §3 and `NORMNET.md §6`).

---

## 2. Implementation — mostly already exists

Take the general case: train a fresh net, then prune. Five steps; **steps 3–4 only for the
pruning case; only step 3 is genuinely new, and it is small.**

| # | Step | What it needs | Status |
|---|------|---------------|--------|
| 1 | Normal train | Standard SGD recipe | reuse |
| 2 | Normalize + fold (reparam) | Insert BN(affine=F), fold μ→bias, σ→weight (`ṽ=σW`) | reuse (`reparam.py`) |
| 3 | Regularize | push contribution `‖ṽ‖→0` | **new, simple** |
| 4 | Prune | score channels, cut, recalibrate | score **new (ours)**; cut/recalib reuse |
| 5 | Fine-tune | Standard SGD recovery | reuse |

**Step 3 (new).** Two regularizer options from the report:
- *Option 2 (normalize columns, train `Ŵ_l = Σ_l W_l`):* this **is native** to our sparse
  training — plain weight decay on `ṽ` already equals contribution decay, and the explicit
  group-lasso `regularization_loss()` is in place. Nothing new to derive.
- *Option 1 (regularize rows of `W_{l-1}` and columns of `W_l` together):* also simple —
  a group penalty over the two coupled weight slabs; no normalization machinery needed.

So step 3 is a thin penalty term, not a subsystem.

**Step 4 (criterion).** The established score is the **per-layer** magnitude
`‖ṽ‖ = ‖σW‖ = √NCI` (degenerate one-layer case). The new contribution is the **propagated**
score — the recursive `Iˡ = W̄ˡ … W̄ᴸ Iᵒ` with the output seed `Iᵒ` handled explicitly.
σ is folded into `M = σ·W = W'` **once**; there is **no separate per-hop σ transfer** — the
PDF's interleaved `Σˡ⁺¹` is already absorbed into the next layer's `Mˡ⁺¹ = σ_inˡ⁺¹·W` (steps
7–10). `relative` selects only the **column normalizer** `D`:
- **relative** → `W̄ = Mᵖ / Σ_i Mᵖ` (column-stochastic, `ρ=1`) — within-layer / local.
- **non-relative** → `W̄ = Mᵖ / σ_pre`ᵖ`, σ_pre = (Σ_i M²)^{1/2}` (std / L2 col-norm,
  steps 7–8) — cross-layer in intent.

**For `p=2` the two normalizers COINCIDE** (`Σ_i M² = (Σ_i M²)^{2/2}`): the PDF states
"variance propagation yields the same relative importance criterion for p=2." They differ only
at `p=1`. The residual-join weighting `σᵖ/Σσᵖ` is a separate fan-in split
(`build_propagation_topology`), orthogonal to `D`.

*(Errata: earlier drafts stated non-relative `W̄ = Mᵖ` raw, then a column-stochastic
`W̄ × σ_post`ᵖ`** transfer — **both WRONG**. The `σ_post` transfer **double-counts** σ: since
`σ_post`ˡ` = σ_inˡ⁺¹` is already inside `Mˡ⁺¹`, re-multiplying applies σ twice per hop (an extra
`σ²` per layer that compounds with depth). The PDF form has σ once and **no transfer**; the
rel/nonrel split is the `D` norm (L1 vs L2). No-free-lunch: column-stochastic (rel, `p=2`) is
bounded but conserves mass → every layer sums to the same total → cross-layer ranking is only
`1/width`, i.e. local; non-stochastic (`p=1`) lets totals differ but `ρ≠1` → per-layer scale
grows `ρ^depth` → a global sort degenerates to ranking-by-depth. Neither propagated form is a
bounded global criterion — the bounded global score is `√NCI` (one hop, no iteration).)*

---

## 3. Optimization-geometry view of the transform

The transform earns its keep in plain training too. Centering and rescaling each layer's
input preconditions the SGD landscape — which is why it pays to apply it at every layer, not
just the network input. The idea traces back to LeCun (*Efficient BackProp*, 1998); we recap
it here to explain why the reparam also speeds optimization.

**Setup.** Take one linear unit `y = Wx` trained by SGD. Near the loss-minimizing weight
`W*`, the loss is locally quadratic, `L(W) ≈ ½ (W−W*)ᵀ H (W−W*)` (second-order Taylor expansion), where the Hessian
`H = E[x xᵀ]` is the input second-moment matrix. SGD descends fast when this bowl is round
and zig-zags when it is elongated. The elongation is the **condition number**
`κ = λ_max/λ_min` — the ratio of largest to smallest curvature, where `λ` are the eigenvalues
of `H`. Large κ → slow: the iterate error contracts by a factor `(κ−1)/(κ+1)` per step
(Nesterov 2004, §2.1; Boyd & Vandenberghe 2004, §9.3.1).

Split the input into mean + fluctuation, `x = μ + x̃`, giving `H = μμᵀ + Σ` (`Σ = Cov(x)`,
diagonal entries `σ_i²`). Two things inflate κ, and the transform removes each:

1. **The mean.** `μμᵀ` is rank-1 with one large eigenvalue `≈ ‖μ‖²` — a single steep
   direction that stretches the bowl. **Folding μ into the bias** leaves the trainable weight
   acting on the centered `x̃`, so its curvature is `Σ` alone and that stretch is gone.
   (Equivalently, the gradient `∂L/∂W = δμᵀ + δx̃ᵀ` loses the rank-1 `δμᵀ` term that pulls
   every output row by the shared mean.)
2. **Unequal channel scales.** `Σ`'s diagonal `σ_i²` differs across channels → steep along
   high-variance inputs, flat along low ones. **Dividing by σ** flattens the diagonal toward
   unit, equalizing the eigenvalues → `κ → 1`.

**Tie to our work.** This *is* the transform: the trainable `ṽ = σW` acts on the normalized
input `x̃ = (x−μ)/σ`, so **training runs in coordinates where `H ≈ I`, at every layer**, and
weight decay then acts on `ṽ` in that well-conditioned space instead of on raw `W`. The one
reparam that turns weight magnitude into the contribution score (pruning) also rounds the
optimization bowl (training).

**Numeric.** Input `x ∈ ℝ²`, uncorrelated, `μ = [5, 5]`, `σ = [1, 0.2]`:

```
raw         H = μμᵀ+Σ = [[26,25],[25,25.04]]   λ = {50.5, 0.515}   κ ≈ 98
fold mean   H = Σ      = diag(1, 0.04)          λ = {1, 0.04}       κ = 25
+ normalize H = I      = diag(1, 1)             λ = {1, 1}          κ = 1
```

`κ: 98 → 25 → 1` — folding the mean kills the rank-1 stretch, normalizing flattens the rest.

The basis is shared with LeCun, but his version enforces it through the choice of activation
function (symmetric activations that keep each layer's output near zero-mean, unit-variance);
this work bakes it into the network directly, normalizing every unit by its own measured
statistics.

**Why it differs, and why it matters here.** BN and LeCun shape the *activations*: BN
normalizes the forward signal — killing the input's per-channel stats, then optionally
recovering them through learnable γ, β — so it is **not** function-preserving. This work
applies the same normalization but **folds it into the weight** (`ṽ = σW`), leaving the
forward unchanged (function-preserving): the object it shapes is the **weight**, not the
signal. That is what makes the weight magnitude `‖σW‖` the contribution score — a
training-time conditioning trick becomes the pruning criterion and its regularizer.

---

## 4. Code — normalize & fold

The fold is in **`torch_pruning/utils/reparam.py`**, in
`NormalizedResidualManager._make_reparam` (build) and `BNResidualLinear.forward`
(function-preserving forward). *(Note: lines ~1250–1350 are the §4 criterion —
`propagation_importance` — not the fold.)*

**Build (the fold), `_make_reparam`, Linear branch:**

```python
W = module.weight.data.clone()                 # [out, in]
b = module.bias.data.clone() if module.bias is not None else torch.zeros(...)
sigma_eff = torch.sqrt(bn_running_var + BN_EPS) # σ from calibrated BN stats
v_tilde = W * sigma_eff[None, :]   # FOLD σ INTO WEIGHT: ṽ = σ·W (now acts on unit-var input)
m       = b + W @ mu_x             # FOLD μ INTO BIAS:   m = b + W·μ
```

**Forward (un-fold via effective bias — no BN materialized), `BNResidualLinear.forward`:**

```python
sigma = torch.sqrt(self.bn.running_var + self.bn.eps)
mu    = self.bn.running_mean
w_eff    = self.v_tilde / sigma[None, :]   # ṽ/σ = W
eff_bias = self.m - w_eff @ mu             # m − W·μ = b
return F.linear(x, w_eff, eff_bias)        # ≡ ṽ·(x−μ)/σ + m
```

Recommend showing **these two snippets** (Linear variant — cleaner than Conv2d). Conv2d is
identical with `sum(dim=(2,3))` over the kernel and `[None,:,None,None]` broadcasting.
`‖ṽ[:,k]‖ = √NCI` is the **per-layer** prune score — plain magnitude on the normalized weight.

### The propagated score `I` (step 4)

Same file, `propagation_importance` (sequential walk) + `_propagate_dag` (residual DAG). Per
layer build `M[in,out] = |σ_i · reduce_kernel(ṽ[out,in,·])|`, then recurse `I^l = W̄^l I^{l+1}`:

```python
# per-layer matrix M[in, out] = |σ_i · reduce(ṽ[out, in, ·])| = |W'|  (σ folded in ONCE)
w = _contribution_weight(rp).detach()          # ṽ = σ·W  ([out,in] or [out,in,kH,kW])
w_red = w.flatten(2).norm(p=2, dim=2) if w.dim()==4 else w   # collapse kernel → [out,in]
M = w_red.t().abs()                            # [in, out]

# recursion:  I^l = W̄^l · I^{l+1}   (PDF steps 7-10; NO separate σ transfer)
Mp = M.pow(p)                                  # p=2 variance / p=1 std
if relative:                                   # column-stochastic (L1) → within-layer
    denom = Mp.sum(dim=0).clamp(min=eps)
else:                                          # σ_pre^p (L2 col-norm) → cross-layer (steps 7-8)
    denom = M.pow(2).sum(dim=0).clamp(min=eps).pow(p / 2.0)
Wbar = Mp / denom[None, :]
I_l = Wbar @ I_next                            # propagate one layer back → [in]   (σ once)
```

For `p=2` the two `denom` branches are equal → relative ≡ non-relative. Seed: `I_next = I_out`
(uniform `1/out` by default) at the last/terminal layer; residual joins sum downstream `I`
weighted by `σᵖ/Σσᵖ` (`build_propagation_topology`). The walk runs in reverse layer order and
returns `OrderedDict[name → I^l]` in forward order.
