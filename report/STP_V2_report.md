---
title: "SIRC Torch-Pruning v2 (STP v2)"
subtitle: "Internal Technical Report"
author:
  - Avraham Raviv
date: "February 2026"
abstract: |
  Fork of Torch-Pruning extended with Pruning-Aware Training (PAT),
  new importance criteria (VBP, MAC-Aware), bias compensation, and regularization strategies.

  \medskip
  \noindent\textbf{Repository:} \texttt{sirc-torch-pruning} (\texttt{master}) \quad
  \textbf{Upstream:} Torch-Pruning v1.6.1 \\
  \textbf{References:} DepGraph [Fang et al., CVPR 2023], VBP [arXiv 2507.12988]
geometry: margin=2.5cm
fontsize: 11pt
documentclass: article
toc: true
toc-depth: 3
numbersections: true
header-includes:
  - \usepackage{graphicx}
  - \usepackage{amsmath,amssymb}
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{xcolor}
  - \definecolor{linkblue}{HTML}{1565C0}
  - \usepackage[colorlinks=true,linkcolor=linkblue,urlcolor=linkblue,citecolor=linkblue]{hyperref}
  - \usepackage{enumitem}
  - \setlist{nosep}
  - \usepackage{caption}
  - \captionsetup{font=small,labelfont=bf}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{\small STP v2 --- Internal Technical Report}
  - \fancyhead[R]{\small SIRC}
  - \fancyfoot[C]{\thepage}
  - \usepackage{listings}
  - \lstset{basicstyle=\small\ttfamily, breaklines=true, frame=single, backgroundcolor=\color{gray!5}}
---

\newpage

# Theoretical Foundations

This part presents the theoretical framework behind structured channel pruning in STP v2. We begin with the dependency-aware pruning mechanism, then introduce Pruning-Aware Training (PAT) --- the main contribution --- followed by pluggable importance criteria, regularization, and post-pruning compensation.


## Structured Pruning Mechanism

### Overview

Structured pruning removes entire channels (or filters) from a network, reducing both parameter count and computational cost. Unlike unstructured (weight-level) pruning, structured pruning produces dense models that run efficiently on standard hardware without specialized sparse kernels.

The pruning engine follows five core steps plus an optional compensation step (Figure~\ref{fig:pruner-arch}):

\begin{figure}[H]
\centering
\includegraphics[width=0.7\textwidth]{figures/pruner_arch.png}
\caption{Pruning pipeline. Five core steps from pretrained model to pruned model, plus an optional compensation step for criteria that collect activation statistics. The dashed loop indicates PAT iterations (Section~\ref{sec:pat}). Criteria highlighted in bold are new in STP~v2.}
\label{fig:pruner-arch}
\end{figure}

\noindent\textbf{Step 1: Build DependencyGraph.} \texttt{DependencyGraph.build\_dependency(model, example\_inputs)} traces a dummy forward pass and records how layers are structurally coupled: if a Conv2d's output channels are pruned, the downstream BatchNorm and the next Conv2d's input channels must be pruned in lockstep.

\noindent\textbf{Step 2: Enumerate Pruning Groups.} \texttt{DG.get\_all\_groups()} yields \texttt{Group} objects --- sets of coupled (layer, channel-indices) pairs that must be pruned together. This is the core contribution of DepGraph [Fang et al., CVPR 2023]: automatic dependency discovery replaces manual group definitions.

\noindent\textbf{Step 3: Score Channels.} The importance criterion's \texttt{\_\_call\_\_(group)} method returns a score vector $\mathbb{R}^C$. The criterion is pluggable --- \texttt{GroupMagnitudeImportance}, \texttt{VarianceImportance}, or any custom subclass can be swapped at initialization (Section~\ref{sec:criteria}).

\noindent\textbf{Step 4: Select Channels.} Channels below a threshold are marked for removal. The threshold can be local (per-layer ratio) or global (single threshold across all groups). Steps 2--4 are orchestrated by \texttt{BasePruner.\_prune()}, which is called from \texttt{BasePruner.step()}.

\noindent\textbf{Step 4b: Compensate} *(optional)*. When a \texttt{mean\_dict} is provided (activation means from a calibration pass), \texttt{BasePruner.\_apply\_compensation(group, idxs)} corrects consumer biases before the structural cut (Section~\ref{sec:compensation}). This is criterion-agnostic --- any pruner (magnitude, LAMP, VBP) can opt in. Without \texttt{mean\_dict}, this step is skipped.

\noindent\textbf{Step 5: Structural Removal.} \texttt{Group.prune(idxs)} removes weight rows/columns, adjusts biases, and updates normalization parameters atomically across all coupled layers.

In PAT mode (Section~\ref{sec:pat}), steps 2--5 repeat $N$ times with interleaved training between steps. The \texttt{ChannelPruning} wrapper (Section~3.2) calls \texttt{step()} at the right epochs automatically.


## Pruning-Aware Training (PAT)
\label{sec:pat}

PAT is the central contribution of STP v2. It wraps the single-pass pruning engine (Section~1.1) behind a thin interface and integrates it into a standard training loop, enabling iterative prune-then-train cycles that preserve accuracy far better than one-shot pruning.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/pat_pipeline.png}
\caption{PAT pipeline. Three phases (all optional), with a repeated pruning loop in Phase~2. Grey rows in Phase~2 are criterion-specific (apply only to VBP or other statistics-based criteria); the blue and yellow rows are shared by all criteria. One-shot mode collapses Phase~2 to a single step with $M = 0$.}
\label{fig:pat-pipeline}
\end{figure}

\noindent\textbf{Phase 1: Sparse Pre-training} *(optional).* Before any structural pruning, the model is trained with a group-sparsity regularizer ($L_{2,1}$ or GMP) that pushes unimportant channels toward zero. This primes the importance signal so that the pruner can make better decisions in Phase~2. Skipping this phase is valid but may reduce pruning quality at aggressive ratios.

\noindent\textbf{Phase 2: Iterative Pruning.} The core PAT loop, repeated $N$ times:

\begin{enumerate}[leftmargin=2em]
\item \textit{Collect stats} \textcolor{gray}{(optional)} --- criteria that rely on activation statistics (e.g., VBP) run a calibration forward pass to gather per-channel means and variances. Weight-only criteria (magnitude, LAMP) skip this.
\item \textit{Score \& prune channels} --- the importance criterion scores every channel, a threshold selects the bottom fraction, and the pruner structurally removes them (Section~1.1, steps 3--5).
\item \textit{Compensate} \textcolor{gray}{(optional)} --- VBP applies bias correction and BN variance update before the structural cut (Section~\ref{sec:compensation}). Other criteria skip this.
\item \textit{Train $M$ epochs} --- standard training (cross-entropy, optionally knowledge distillation) to recover accuracy before the next pruning step.
\end{enumerate}

\noindent Each step retains a fraction $q = r^{1/N}$ of channels, so after $N$ steps the cumulative retention is $q^N = r$. One-shot mode is the degenerate case: $N = 1$, $M = 0$.

\noindent\textbf{Phase 3: Fine-tuning.} After all pruning steps, the pruned model is trained for several more epochs with no further pruning. For CNNs, BN running statistics are recalibrated at the start of this phase (Section~1.5.3).

### Pruning Schedules

Rather than reimplementing a schedule externally, PAT leverages the upstream \texttt{BasePruner}'s built-in \texttt{iterative\_steps} parameter. The pruner is created once with a target ratio and $N$ iterative steps; each call to \texttt{step()} applies the next increment automatically. The schedule determines how the target ratio $r$ is distributed across these $N$ steps:

\noindent\textbf{Linear} (upstream default). The target pruning ratio $p = 1 - r$ is divided equally across $N$ steps: step $t$ prunes to cumulative ratio $p \cdot t / N$. Simple, but early steps are disproportionately aggressive --- removing $p/N$ channels from a full model is a smaller relative cut than removing $p/N$ from an already-pruned one.

\noindent\textbf{Geometric.} Each step retains a constant fraction $q$ of the \textit{current} channels, so that the cumulative effect equals the target:
$$q = r^{1/N}, \qquad q^N = r$$
This ensures equal \textit{proportional} pruning at each step, making each cut equally disruptive relative to the remaining capacity. Preferred over linear for aggressive ratios.

\noindent\textbf{One-shot.} The degenerate case of either schedule with $N = 1$: no interleaved training ($M = 0$), followed by fine-tuning only. Simpler and faster, but less accurate at aggressive ratios.

### Training Loop Integration

The PAT wrapper exposes two calls --- \texttt{regularize()} and \texttt{prune()} --- to an otherwise unchanged training loop. Based on the current epoch and configuration (\texttt{start\_epoch}, \texttt{end\_epoch}, \texttt{epoch\_rate}), the wrapper decides internally whether to apply regularization, trigger a pruning step, or do nothing (fine-tuning phase). The chosen schedule (linear or geometric) only affects the ratio applied at each step; the epoch-based control is the same regardless.


## Channel Importance Criteria
\label{sec:criteria}

### Group-Aware Importance (DepGraph Contribution)

Prior work [Li et al., ICLR 2017] computed importance per-layer: $I_c = \|\mathbf{w}_c\|_p$ for each layer independently. This ignores inter-layer dependencies --- pruning channel $c$ from a Conv2d also affects the downstream BN and the next layer's input, but per-layer scoring does not account for this.

DepGraph introduced **group-level importance aggregation**. For a pruning group containing $K$ coupled layers, each layer $k$ contributes a local importance vector $\mathbf{i}^{(k)} \in \mathbb{R}^C$. These are aggregated via a configurable reduction over a shared \textit{root channel space} (mapped via the DG's index tracking):

$$I_c = \text{reduce}\!\left(\left\{ i_c^{(1)},\, i_c^{(2)},\, \ldots,\, i_c^{(K)} \right\}\right)$$

Available reductions include \textbf{mean} (default, $I_c = \frac{1}{K}\sum_k i_c^{(k)}$), \textbf{max}, \textbf{prod}, and \textbf{first}/\textbf{gate} (use only one layer in the group). This group reduction is orthogonal to the base criterion --- it applies equally to magnitude, Taylor, Hessian, and VBP importance.

### Magnitude-Based Importance

The base criterion computes per-channel weight norms. For layer $m$ with weight tensor $W^{(m)}$:

$$i_c^{(m)} = \left\| \mathbf{w}_c^{(m)} \right\|_p$$

where $\mathbf{w}_c$ is the $c$-th output channel flattened to a vector ($\mathbb{R}^{C_{\text{in}}}$ for Linear, $\mathbb{R}^{C_{\text{in}} \cdot k^2}$ for Conv2d). Input-channel pruning transposes the weight before computing norms. BatchNorm/LayerNorm affine weights contribute their scale parameters.

With group aggregation (Section 1.3.1), this becomes \texttt{GroupMagnitudeImportance}: each coupled layer contributes its local magnitude, and the group reduction produces a single consensus score per channel. This is fast (no data required), but ignores activation statistics and gradient information.

Post-reduction normalization can further adjust scores: per-layer mean normalization, max normalization, Gaussian z-scores, or LAMP (Layer-Adaptive Magnitude Pruning) weighting.

### Variance-Based Pruning (VBP)

VBP [arXiv 2507.12988] uses the variance of **post-activation outputs** as importance. A channel whose activations have near-zero variance produces a nearly constant signal, contributing little information to downstream layers.

**Statistics collection.** Forward hooks on target layers accumulate running sums $\sum x_c$ and $\sum x_c^2$ across a calibration set. After $N$ total observations:

$$\mu_c = \frac{1}{N}\sum x_c, \qquad \sigma_c^2 = \frac{1}{N}\sum x_c^2 - \mu_c^2$$

The importance score for channel $c$ is $I_c = \sigma_c^2$. Lower variance $\to$ lower importance $\to$ pruned first. The stored means $\mu_c$ are also used for bias compensation (Section~\ref{sec:compensation}).

**Spatial reduction.** For Conv2d feature maps $(B, C, H, W)$, each spatial position is an independent observation ($N = B \cdot H \cdot W$). For Linear outputs $(B, T, C)$, sequence positions play the same role ($N = B \cdot T$). The kernel size and spatial resolution affect only the sample count, not the dimensionality of the score.

**Architecture-specific hooking.** Statistics are collected after the activation function (post-GELU for ViT fc1 layers, post-BN+ReLU for CNN convolutions) via composable \texttt{post\_act\_fn} hooks.

### MAC-Aware Importance

MAC-Aware importance is designed as a **composite criterion** that balances structural importance (accuracy) against computational cost (efficiency). The general formulation combines an accuracy-driven term with a cost-weighted term:

$$I_c = \alpha \cdot \left\| I_c^{\text{base}} \right\| + (1 - \alpha) \cdot \left(\frac{\text{cost}_m}{\text{cost}_{\max}}\right)^{\!\beta} \qquad \text{(additive mode)}$$

$$I_c = I_c^{\text{base}} \cdot \left(\frac{\text{cost}_m}{\text{cost}_{\max}}\right)^{\!\beta} \qquad \text{(multiplicative mode)}$$

Parameters $\alpha$ (importance vs.\ cost trade-off) and $\beta$ (cost exponent) control the balance.

**Current implementation** uses MACs as the cost metric, but the cost term is intentionally designed to be replaceable. In deployment scenarios, MACs can be substituted with measured GPU latency, power consumption, memory bandwidth, or any hardware-specific PPA (Power--Performance--Area) metric --- the formulation remains the same. This makes the criterion hardware-agnostic: optimize for whatever efficiency metric matters for the target platform.


## Regularization for Structured Pruning

Structured pruning removes entire channels. For this to work well, the model should be trained so that unimportant channels are clearly distinguishable --- ideally near-zero. Regularization during training achieves this by adding a penalty term to the loss.

### DepGraph Group Regularization

DepGraph [Fang et al., CVPR 2023, Section 3.3, Eq.~4--5] defines an $L_{2,1}$ norm over pruning groups. For group $G_k$ containing all coupled layers, let $\mathbf{g}_k^{(i)}$ be the concatenated parameters of the $i$-th channel across all layers in the group:

$$\mathcal{L}_{\text{reg}} = \sum_k \sum_{i=1}^{C_k} \left\| \mathbf{g}_k^{(i)} \right\|_2$$

This is $L_2$ within each channel (preserving internal structure), $L_1$ across channels (encouraging entire channels to zero) --- the standard *group sparsity* formulation.

### $L_{2,1}$ Weight Regularization

Our implementation applies the same $L_{2,1}$ norm, restricted to the prunable layers (MLP expansion or interior convolutions):

$$\mathcal{L}_{L_{2,1}} = \sum_{m} \sum_{i=1}^{C_{\text{out}}} \left\| \mathbf{w}_i^{(m)} \right\|_2$$

where $\mathbf{w}_i^{(m)}$ is the $i$-th output channel of layer $m$, flattened to a vector. For a Linear layer, $\mathbf{w}_i \in \mathbb{R}^{C_{\text{in}}}$; for a Conv2d layer with kernel size $k$, $\mathbf{w}_i \in \mathbb{R}^{C_{\text{in}} \cdot k^2}$. The total loss becomes:

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{L_{2,1}}$$

This operates in **weight space** and is agnostic to the pruning criterion.

### Variance Entropy Regularization

Designed specifically for VBP. Instead of penalizing weights, it shapes the **activation variance distribution** to sharpen the VBP importance signal.

For each target layer $m$ with output activations $\mathbf{a}$, compute the per-channel variance $\sigma_c^2$ (across batch and spatial/sequence dimensions), normalize into a probability distribution, and compute its entropy:

$$p_c = \frac{\sigma_c^2}{\sum_j \sigma_j^2}, \qquad H_m = -\sum_c p_c \log p_c, \qquad \mathcal{L}_{\text{var}} = \sum_m H_m$$

Entropy is maximal when all channels have equal variance (uniform $p$) and minimal when variance is concentrated in few channels. **Minimizing** $\mathcal{L}_{\text{var}}$ forces the model to concentrate its representational capacity into fewer channels, making the remaining low-variance channels clearly expendable.

### Comparison

\begin{table}[H]
\centering
\caption{Comparison of regularization strategies.}
\begin{tabular}{lll}
\toprule
& \textbf{$L_{2,1}$ Weight Regularization} & \textbf{Variance Entropy Regularization} \\
\midrule
Domain & Weight space & Activation space \\
Mechanism & Group sparsity (push channels $\to 0$) & Entropy minimization (concentrate variance) \\
Alignment & Agnostic (benefits magnitude pruning) & Directly aligned with VBP \\
Phase & Sparse pre-training (before pruning) & PAT fine-tuning (during pruning) \\
\bottomrule
\end{tabular}
\label{tab:reg-comparison}
\end{table}


## Post-Pruning Compensation
\label{sec:compensation}

### Bias Compensation

When channel $j$ is pruned from a layer's output, every downstream consumer loses that channel's contribution. For a consumer computing $y_i = \sum_j W_{ij}\, x_j + b_i$, the output after pruning channel set $P$ becomes:

$$y_i^{\text{pruned}} = \sum_{j \notin P} W_{ij}\, x_j + b_i$$

The missing term has expected value $\sum_{j \in P} W_{ij}\, \mu_j$, where $\mu_j = \mathbb{E}[x_j]$ is the mean activation of the pruned channel (collected via a calibration forward pass). Compensation absorbs this into the bias:

$$b_i^{\text{new}} = b_i + \sum_{j \in P} W_{ij}\, \mu_j$$

This is implemented in \texttt{BasePruner.\_apply\_compensation()} and is **criterion-agnostic**: any pruner (magnitude, LAMP, Taylor, VBP) can opt in by passing a \texttt{mean\_dict} mapping modules to per-channel activation means. A standalone \texttt{collect\_activation\_means()} utility collects these means from a calibration set without requiring VBP's full variance statistics. When no \texttt{mean\_dict} is provided, compensation is skipped with zero overhead.

This restores the correct **expected output**. However, the **variance** is not restored --- the fluctuations $\sum_{j \in P} W_{ij}(x_j - \mu_j)$ are permanently lost:

$$\operatorname{Var}\!\left(y_i^{\text{pruned}}\right) = \operatorname{Var}(y_i) - \sum_{j \in P} W_{ij}^2\, \sigma_j^2$$

(assuming independent channels). This variance shift is the root cause of the CNN-specific normalization problem.

### The BatchNorm Problem in CNNs

**Why ViTs are resilient.** In a ViT, every Linear layer is followed by **LayerNorm**, which computes normalization statistics from the current input at inference time:

$$\hat{x}_c = \frac{x_c - \mu_{\scriptscriptstyle\text{LN}}}{\sigma_{\scriptscriptstyle\text{LN}}}, \qquad \mu_{\scriptscriptstyle\text{LN}} = \frac{1}{C}\sum_c x_c, \quad \sigma_{\scriptscriptstyle\text{LN}}^2 = \frac{1}{C}\sum_c (x_c - \mu_{\scriptscriptstyle\text{LN}})^2$$

After pruning and bias compensation, the mean is corrected and the variance has shifted --- but LayerNorm recomputes $\mu_{\scriptscriptstyle\text{LN}}$ and $\sigma_{\scriptscriptstyle\text{LN}}$ from whatever the pruned network actually produces. It **automatically adapts**.

**Why CNNs break.** In a CNN, Conv2d layers are followed by **BatchNorm**, which normalizes using **stored population statistics** accumulated during the original training:

$$\hat{x}_c = \frac{x_c - \hat{\mu}_c}{\sqrt{\hat{\sigma}_c^2 + \epsilon}}$$

where $\hat{\mu}_c$ and $\hat{\sigma}_c^2$ are exponential moving averages frozen at inference. After pruning:

1. $\hat{\mu}_c$ **is stale** --- bias compensation corrects the consumer's bias, but BN sits between producer and consumer. The actual per-channel mean has changed; BN still subtracts the old mean.
2. $\hat{\sigma}_c^2$ **is stale** --- the variance reduction from pruning means the true activation variance is now smaller, but BN divides by the old (larger) standard deviation, *compressing* the signal.
3. **Errors compound** --- each BN layer introduces a mean/variance mismatch. In a deep network these errors accumulate through the residual stream, leading to catastrophic accuracy collapse.

This is not specific to VBP --- *any* structured pruning criterion applied to CNNs suffers from stale BN statistics.

\begin{figure}[H]
\centering
\includegraphics[width=0.62\textwidth]{figures/bn_vs_ln.png}
\caption{Normalization after pruning. \textbf{Left:} ViT with LayerNorm --- statistics are recomputed on-the-fly, adapting automatically. \textbf{Right:} CNN with BatchNorm --- stored running statistics become stale, requiring explicit recalibration.}
\label{fig:bn-vs-ln}
\end{figure}

### BN Recalibration

The fix is empirical recalibration: after pruning, forward a calibration set through the pruned network with BN layers in training mode (recomputing statistics) but no gradient updates:

1. **Reset** all BN running statistics: $\hat{\mu} \leftarrow 0$, \; $\hat{\sigma}^2 \leftarrow 1$.
2. **Forward** $K$ calibration batches in training mode (no backpropagation).
3. BN layers accumulate fresh $\hat{\mu}$ and $\hat{\sigma}^2$ reflecting the pruned network's actual activation distribution.
4. **Freeze** by switching to evaluation mode.

The reset step is critical --- without it, BN's exponential moving average blends new (correct) statistics with old (stale) ones, and convergence is slow. With reset, a few hundred batches suffice for ResNet-50; MobileNetV2's narrow bottleneck layers need ${\sim}5{,}000$ samples.


\newpage

# STP v2 Implementation

This part summarizes the engineering changes in STP v2 relative to the upstream Torch-Pruning library.

## Upstream Rebase

STP was rebased from Torch-Pruning v1.5.1 to **v1.6.1**, gaining a refactored dependency module with cleaner Node/Group/DependencyGraph separation and improved isomorphic and multi-head attention pruning support.

## ChannelPruning Simplification

Major refactor of the pruning scheduler --- both code cleanup and an architectural simplification.

**Code cleanup.** PEP~8 class naming (\texttt{channel\_pruning} $\to$ \texttt{ChannelPruning}), a \texttt{PruningMethod} enum replacing magic strings, a unified logging helper, and bug fixes for \texttt{ignored\_layers} accumulation and hardcoded device references.

**Scheduling rewrite.** Previously, the pruner was recreated every epoch with custom rate computation, MAC-target flags, optimizer resets, and JSON config writing. Now the pruner is created **once at initialization** with a fixed number of iterative steps, and each pruning epoch calls a single \texttt{step()}. The upstream linear scheduler splits the target ratio across steps automatically.

**Regularization cleanup.** The \texttt{regularize()} method was simplified: a single \texttt{epoch > end\_epoch} guard replaces scattered conditionals, and the separate L1 group sparse pre-training code was removed (now handled by the training script). \texttt{GroupNormPruner} now uses \texttt{GroupMagnitudeImportance} internally (fixing an AttributeError with the non-existent \texttt{GroupNormImportance}), and \texttt{ConvTranspose2d} layers are skipped during regularization to avoid shape mismatches.

**Regularization loss visibility.** TP's regularization works by modifying gradients directly (\texttt{param.grad += ...}), not by adding a loss term. To enable separate logging of the regularization magnitude, \texttt{regularize()} now accumulates and returns the $L_1$ norm of the gradient modifications as a scalar. Callers can log this alongside the task loss without changing the optimization.

**MAC-target estimation.** An analytical conversion from MAC retention target to channel keep ratio was added. Each layer's MACs are classified as unpruned ($U$), 1-dimensional ($S_1$, only input or output pruned), or 2-dimensional ($S_2$, both pruned). The target equation:

$$\text{mac\_target} = \frac{U + S_1 \cdot q + S_2 \cdot q^2}{T}, \qquad q = 1 - p \;\text{(keep ratio)}$$

For ViT (all 1-dim): linear solution $p = 1 - \text{mac\_target}$. For CNNs with 2-dim middle layers: quadratic formula. Called once at initialization; the pruner's iterative steps handle the splitting.

**Constructor \texttt{train\_loader} parameter.** The \texttt{Pruning} and \texttt{ChannelPruning} constructors accept an optional \texttt{train\_loader}, enabling statistics collection and compensation mean gathering in a single pass during initialization. A \texttt{\_stats\_fresh} flag prevents redundant re-collection on the first \texttt{prune()} call.

**One-shot / PAT epoch config.** The \texttt{vbp\_imagenet\_pat.py} config builder handles \texttt{pat\_steps $\leq$ 1} as one-shot (\texttt{end\_epoch = start\_epoch}, \texttt{iterative\_steps = 1}) and \texttt{pat\_steps > 1} as PAT (\texttt{end\_epoch = start\_epoch + (pat\_steps - 1) $\times$ epoch\_rate}), ensuring \texttt{iterative\_steps} exactly equals \texttt{pat\_steps}.


## Transformer Support

The original pruning utilities only handled Conv2d and BatchNorm. Extensions include:

- **Linear layer pruning** --- proper in/out channel handling for \texttt{nn.Linear}.
- **LayerNorm pruning** --- coupled with Linear in dependency groups (ViT uses LayerNorm before/after attention and MLP blocks).

**Future work:** MHA head pruning (Q/K/V/output projection simultaneously), embedding dimension pruning, and joint MLP + attention pruning modes.


## Generalized Bias Compensation

Bias compensation was originally implemented inside \texttt{VBPPruner}. It has been **lifted to \texttt{BasePruner}** so that any pruning criterion can benefit from it:

- \texttt{BasePruner.\_apply\_compensation(group, idxs)} handles Linear, Conv2d (standard and depthwise) consumers.
- Activated by passing \texttt{mean\_dict} at construction or via \texttt{set\_mean\_dict()}.
- \texttt{collect\_activation\_means()} provides a standalone calibration utility (no VBP dependency).
- \texttt{ChannelPruning} auto-collects means when a \texttt{train\_loader} is available, regardless of criterion.
- \texttt{VBPPruner} now inherits compensation from \texttt{BasePruner} and only adds mean-check diagnostics and BN variance updates.

### Auto-Detected Target Layers

Target layers for statistics collection are now auto-detected by walking the \texttt{DependencyGraph} rather than requiring architecture-specific code. \texttt{build\_target\_layers()} discovers Conv2d$\to$BN$\to$Act and Linear$\to$Act patterns automatically, supporting CNNs, ViTs, and ConvNeXt with a single code path.

### CNN-Specific Challenges

Tested on ResNet-50 and MobileNetV2. Key challenges resolved:

- **BN recalibration** --- running statistics must be explicitly reset before recalibrating (Section~1.5.3).
- **Depthwise conv group roots** --- in MobileNetV2, the depthwise conv is the group root but activation statistics reside on the expand conv. The compensation routine searches the group for a module with matching statistics.
- **Depthwise convs and ignored layers** --- depthwise convs must not be in the ignored-layers list, as they appear in expand conv groups with output-channel pruning, causing group rejection.
- **ReLU6 detection** --- \texttt{nn.ReLU6} uses \texttt{HardtanhBackward0} internally, requiring an explicit mapping in activation auto-detection.

### Mask-Then-Prune Strategy

In PAT mode, intermediate pruning steps now use **mask-only mode** (zeroing channels but keeping the architecture intact), which allows the optimizer state and learning rate schedule to remain consistent across steps. The **final step** switches to physical structural removal, so fine-tuning operates on the actually smaller model with reduced MACs.

**MAC logging.** Mask-only and physical steps require different MAC measurement. Standard \texttt{count\_ops\_and\_params} counts ops from tensor shapes, so zeroed channels still register as full MACs. Mask steps therefore use \texttt{measure\_macs\_masked\_model()} which detects zeroed channels and subtracts their contribution. The final physical step compares current absolute MACs against the stored original. Both paths report a consistent ratio relative to the unpruned model.

### Config Lifecycle

After the final pruning step, \texttt{ChannelPruning} persists \texttt{is\_prune=False} in the JSON config. On subsequent training runs (e.g., resumed fine-tuning), the constructor detects this flag and early-returns, skipping all pruner initialization: MAC estimation, DependencyGraph construction, statistics collection, and graph visualization. This avoids errors when loading an already-pruned model and reduces startup time.

## Dependency Graph Visualization

Graphviz-based visualization of the DependencyGraph and pruning groups, providing three complementary views:

- **Computational graph** --- data flow through the model.
- **Dependency graph** --- pruning coupling between layers (direct dependencies vs.\ forced shape matches).
- **Combined** --- both overlaid with consistent node layout.

Features: group cluster boxes, color-coded nodes by operation type, multi-group detection, and consistent layout across views. Output formats: PNG, SVG, PDF.


\newpage

# Practical Guide

This part explains how to use the two main entry points for VBP pruning.

## Direct Pipeline: \texttt{vbp\_imagenet.py}

The primary research script. Supports one-shot pruning, iterative PAT, sparse pre-training, knowledge distillation, and multi-criterion benchmarking. All three pipeline phases (sparse $\to$ prune $\to$ fine-tune) are controlled via command-line arguments.

### One-Shot Pruning

The simplest mode: collect statistics, prune once, fine-tune.

```bash
python benchmarks/vbp/vbp_imagenet.py \
  --model_type vit --model_name google/vit-base-patch16-224 \
  --data_path /path/to/imagenet \
  --keep_ratio 0.65 --global_pruning \
  --epochs_ft 10 --disable_ddp
```

When \texttt{--pat} is not set, the script automatically converts to one-shot mode (\texttt{pat\_steps=1}, \texttt{pat\_epochs\_per\_step=0}), followed by \texttt{epochs\_ft} epochs of fine-tuning.

### Iterative PAT

Multiple prune-then-train cycles for better accuracy at aggressive ratios:

```bash
python benchmarks/vbp/vbp_imagenet.py \
  --model_type vit --model_name /path/to/deit_tiny \
  --data_path /path/to/imagenet \
  --keep_ratio 0.65 --global_pruning \
  --pat --pat_steps 5 --pat_epochs_per_step 1 \
  --epochs_ft 10 --disable_ddp
```

Each step keeps $q = 0.65^{1/5} \approx 0.894$ of channels. After 5 steps: $0.894^5 = 0.65$ total retention. Between steps: 1 epoch of fine-tuning. After all steps: 10 epochs of final fine-tuning.

### Sparse Pre-Training

Optional first phase to prime the pruning signal before any structural pruning:

**L2,1 group sparsity** (structured signal):

```bash
python benchmarks/vbp/vbp_imagenet.py \
  --sparse_mode l1_group --epochs_sparse 5 --l1_lambda 1e-4 \
  --keep_ratio 0.65 --global_pruning --epochs_ft 10 --disable_ddp \
  --model_type vit --model_name /path/to/model --data_path /path/to/data
```

**Gradual Magnitude Pruning** (unstructured signal):

```bash
python benchmarks/vbp/vbp_imagenet.py \
  --sparse_mode gmp --epochs_sparse 5 --gmp_target_sparsity 0.5 \
  --keep_ratio 0.65 --global_pruning --epochs_ft 10 --disable_ddp \
  --model_type vit --model_name /path/to/model --data_path /path/to/data
```

### CNN Pruning (ResNet-50)

```bash
python benchmarks/vbp/vbp_imagenet.py \
  --model_type cnn --cnn_arch resnet50 \
  --model_name /path/to/resnet50.pth \
  --data_path /path/to/imagenet \
  --keep_ratio 0.8 --global_pruning \
  --opt_ft sgd --lr_ft 0.01 \
  --epochs_ft 15 --disable_ddp
```

For CNNs, BN recalibration is applied automatically after pruning. Use \texttt{--no\_recalib} to skip (not recommended). SGD with higher learning rate (\texttt{--opt\_ft sgd --lr\_ft 0.01}) is preferred over AdamW for CNN fine-tuning.

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 benchmarks/vbp/vbp_imagenet.py \
  --model_type vit --model_name /path/to/deit_tiny \
  --data_path /path/to/imagenet \
  --keep_ratio 0.65 --global_pruning \
  --pat --pat_steps 5 --pat_epochs_per_step 1 \
  --epochs_ft 10 --use_kd
```

Statistics are collected on rank~0 and broadcast to all ranks. Omit \texttt{--disable\_ddp} when using \texttt{torchrun}.

### Multi-Criterion Comparison

```bash
python benchmarks/vbp/vbp_imagenet.py --criterion variance ...   # VBP, default
python benchmarks/vbp/vbp_imagenet.py --criterion magnitude ...  # baseline
python benchmarks/vbp/vbp_imagenet.py --criterion lamp ...       # baseline
python benchmarks/vbp/vbp_imagenet.py --criterion random ...     # baseline
```

VBP-specific logic (stats collection, compensation, variance loss) is automatically gated on \texttt{--criterion variance}; other criteria use the base pruner.


## Pruning Class API: \texttt{vbp\_imagenet\_pat.py}

A thin wrapper demonstrating the \texttt{ChannelPruning} class from \texttt{pruning\_utils.py}. This is the **integration path** --- the IP's training loop stays unchanged, and the pruning logic is encapsulated behind two method calls.

### Integration Pattern

The integration requires three additions to an existing training loop:

```python
from torch_pruning.utils.pruning_utils import Pruning

# 1. Initialize (once) — train_loader enables stats + compensation
pruner = Pruning(model, config_folder, device=device,
                 train_loader=train_loader)

# 2. Pre-loop prune (one-shot prunes here; PAT does step 1)
pruner.prune(model, epoch=0, log=logger, mask_only=False)

# 3. Rebuild optimizer (parameters may have changed)
optimizer = AdamW(model.parameters(), lr=lr)

# 4. Training loop
for epoch in range(1, total_epochs + 1):
    train_one_epoch(model, optimizer, ...)
    pruner.prune(model, epoch, log=logger, mask_only=False)
    pruner.channel_regularize(model)
```

The \texttt{Pruning} constructor accepts \texttt{train\_loader} so that statistics and compensation means are collected during initialization (avoiding a redundant second pass). The pre-loop prune fires at epoch~0; subsequent calls use the internal epoch schedule to decide whether to prune or no-op. A \texttt{\_stats\_fresh} flag ensures statistics are not re-collected on the first pruning call when they were just gathered at init.

### Epoch-Based Scheduling

The training loop uses a 1-indexed convention: a pre-loop call handles epoch~0, then the loop runs from epoch~1 to \texttt{total\_epochs}. The \texttt{ChannelPruning.prune()} method decides what to do at each epoch based on the configuration:

\begin{table}[H]
\centering
\caption{Epoch-based behavior of the ChannelPruning wrapper.}
\begin{tabular}{lll}
\toprule
\textbf{Epoch range} & \textbf{regularize()} & \textbf{prune()} \\
\midrule
Pre-loop (epoch 0) & --- & First prune step (one-shot completes here) \\
Before \texttt{start\_epoch} & Apply L$_{2,1}$ / GMP & No-op (or GMP mask update) \\
Pruning epochs & Apply L$_{2,1}$ / GMP & Re-collect stats, prune step, BN recalib \\
After \texttt{end\_epoch} & No-op & No-op (fine-tuning) \\
\bottomrule
\end{tabular}
\label{tab:epoch-schedule}
\end{table}

After the final pruning step, retention accuracy is logged automatically (post-prune, pre-fine-tuning). The \texttt{prune\_channels} flag transitions to \texttt{False}, and all subsequent \texttt{prune()} calls return immediately.

### Example: PAT with Sparse Pre-Training

```bash
torchrun --nproc_per_node=4 benchmarks/vbp/vbp_imagenet_pat.py \
  --model_type vit --model_name /path/to/deit_tiny \
  --data_path /path/to/imagenet \
  --keep_ratio 0.65 --global_pruning \
  --pat_steps 5 --epochs 15 \
  --pruning_schedule geometric \
  --sparse_mode l1_group --epochs_sparse 5 --l1_lambda 1e-4 \
  --var_loss_weight 0.1 --use_kd
```


## Configuration Reference

\begin{table}[H]
\centering
\caption{Key command-line arguments for \texttt{vbp\_imagenet.py}.}
\small
\begin{tabular}{llll}
\toprule
\textbf{Argument} & \textbf{Default} & \textbf{Type} & \textbf{Description} \\
\midrule
\multicolumn{4}{l}{\textit{Model}} \\
\texttt{--model\_type} & vit & str & Architecture: vit, convnext, cnn \\
\texttt{--model\_name} & --- & str & HuggingFace ID or local path \\
\texttt{--cnn\_arch} & resnet50 & str & CNN variant (if model\_type=cnn) \\
\midrule
\multicolumn{4}{l}{\textit{Pruning}} \\
\texttt{--keep\_ratio} & 0.65 & float & Target channel retention ratio \\
\texttt{--global\_pruning} & False & flag & Global vs.\ per-layer threshold \\
\texttt{--criterion} & variance & str & variance, magnitude, lamp, random \\
\texttt{--no\_compensation} & False & flag & Disable VBP bias compensation \\
\midrule
\multicolumn{4}{l}{\textit{PAT}} \\
\texttt{--pat} & False & flag & Enable iterative PAT \\
\texttt{--pat\_steps} & 5 & int & Number of prune-train cycles \\
\texttt{--pat\_epochs\_per\_step} & 3 & int & FT epochs between prune steps \\
\texttt{--var\_loss\_weight} & 0.0 & float & Variance entropy loss weight \\
\midrule
\multicolumn{4}{l}{\textit{Sparse Pre-training}} \\
\texttt{--sparse\_mode} & none & str & l1\_group, gmp, or none \\
\texttt{--epochs\_sparse} & 5 & int & Sparse training epochs \\
\texttt{--l1\_lambda} & 1e-4 & float & L$_{2,1}$ regularization strength \\
\midrule
\multicolumn{4}{l}{\textit{Fine-tuning}} \\
\texttt{--epochs\_ft} & 10 & int & Post-prune fine-tuning epochs \\
\texttt{--lr\_ft} & 1.5e-5 & float & Fine-tuning learning rate \\
\texttt{--opt\_ft} & adamw & str & Optimizer: adamw or sgd \\
\texttt{--use\_kd} & False & flag & Knowledge distillation \\
\texttt{--kd\_alpha} & 0.7 & float & CE weight in KD loss \\
\midrule
\multicolumn{4}{l}{\textit{Data \& Distributed}} \\
\texttt{--data\_path} & --- & str & ImageNet root (train/val subdirs) \\
\texttt{--train\_batch\_size} & 64 & int & Batch size per GPU \\
\texttt{--max\_batches} & 200 & int & Batches for stats collection \\
\texttt{--disable\_ddp} & False & flag & Single-GPU mode \\
\bottomrule
\end{tabular}
\label{tab:args}
\end{table}


## Common Pitfalls

\begin{enumerate}
\item \textbf{Missing statistics.} If \texttt{collect\_and\_sync\_stats()} is skipped or fails silently, all importance scores default to 1.0. The global threshold then selects \textit{all} channels for pruning, which the validity check rejects --- resulting in 0 groups pruned and no error message. Always verify logs show ``Statistics collected for N layers.''

\item \textbf{Skipping BN recalibration on CNNs.} Without resetting and recomputing BN running statistics after pruning, accuracy drops to ${\sim}0\%$ regardless of pruning ratio or criterion. This is the single most common failure mode for CNN pruning.

\item \textbf{Scheduler granularity.} \texttt{build\_ft\_scheduler()} returns \texttt{(scheduler, step\_per\_batch=True)}. The scheduler must be stepped after every \textit{batch}, not every epoch. Stepping per-epoch with a per-batch scheduler produces near-zero learning rates.

\item \textbf{Variance loss outside PAT.} The \texttt{--var\_loss\_weight} flag only applies during per-step fine-tuning in PAT mode. It has no effect in one-shot mode or during post-prune fine-tuning.

\item \textbf{DDP stat synchronization.} Statistics are collected on rank~0 only and broadcast. If the broadcast fails silently (e.g., process group not initialized), non-main ranks will have empty importance --- leading to inconsistent pruning across GPUs.

\item \textbf{MobileNetV2 one-shot.} One-shot structured pruning is fundamentally broken for MobileNetV2 across all criteria. Always use iterative PAT for narrow-bottleneck architectures.
\end{enumerate}
