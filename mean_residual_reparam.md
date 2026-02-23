---
title: "Mean-Residual Reparameterization for Variance Regularization"
author: "Avraham Raviv"
date: "February 2026"
geometry: margin=2.5cm
fontsize: 11pt
header-includes:
  - \usepackage{amsmath,amssymb,bm}
  - \usepackage{booktabs}
  - \usepackage{enumitem}
  - \usepackage{tikz}
  - \usetikzlibrary{positioning,arrows.meta,calc,fit,backgrounds,decorations.pathreplacing}
  - \setlist{nosep}
---

# Motivation

Magnitude pruning regularizes what it measures: $L_1/L_2$ on weights $\to$ small weights $\to$ prune.
VBP measures activation variance but has **no training-time lever** to push it down.
Weight decay shrinks mean *and* variance indiscriminately.

**Key idea:** Reparameterize into *mean* + *residual*. Regularize only the residual $\Rightarrow$ variance $\to 0$, mean preserved.

# Reparameterization

Standard channel $k$: $\;z_k = \mathbf{w}_k^\top \mathbf{x} + b_k$.
Given fixed $\boldsymbol{\mu}_x = \mathbb{E}[\mathbf{x}]$ from calibration, define:

$$\boxed{z_k = \underbrace{m_k}_{\text{mean}} + \underbrace{\mathbf{v}_k^\top (\mathbf{x} - \boldsymbol{\mu}_x)}_{\text{residual}}}$$

Init: $m_k = \mathbf{w}_k^\top \boldsymbol{\mu}_x + b_k$, $\;\mathbf{v}_k = \mathbf{w}_k$. Same parameter count, identical function at init.

\vspace{0.4cm}

\begin{center}
\begin{tikzpicture}[
    >=Stealth,
    neuron/.style={draw, circle, minimum size=0.55cm, inner sep=0pt, font=\scriptsize},
    op/.style={draw, circle, inner sep=1.5pt, font=\scriptsize},
    label/.style={font=\scriptsize\itshape, text=black!70},
    annot/.style={font=\scriptsize, text=black!60},
]

% === Standard (top) ===
\node[annot, font=\small\bfseries] at (-1.8, 0) {Standard:};

% Input neurons
\node[neuron] (x1) at (0, 0.8) {$x_1$};
\node[neuron] (x2) at (0, 0) {$x_2$};
\node[neuron] (x3) at (0, -0.8) {$x_3$};

% Output neuron (pre-activation)
\node[neuron, minimum size=0.7cm] (z) at (2.5, 0) {$\Sigma$};
\node[annot, above=0.05cm of z] {$b_k$};

% Activation
\node[op] (act) at (3.7, 0) {$\sigma$};
\node[neuron] (ak) at (4.9, 0) {$a_k$};

% Weights
\draw[->, thick] (x1) -- node[above, annot, pos=0.4] {$w_1$} (z);
\draw[->, thick] (x2) -- node[above, annot, pos=0.5] {$w_2$} (z);
\draw[->, thick] (x3) -- node[below, annot, pos=0.4] {$w_3$} (z);
\draw[->] (z) -- node[above, annot] {$z_k$} (act);
\draw[->] (act) -- (ak);

% === Reparameterized (bottom) ===
\node[annot, font=\small\bfseries] at (-1.8, -3.8) {Reparam:};

% Input neurons
\node[neuron] (rx1) at (0, -2.7) {$x_1$};
\node[neuron] (rx2) at (0, -3.5) {$x_2$};
\node[neuron] (rx3) at (0, -4.3) {$x_3$};

% Centering nodes
\node[op] (s1) at (1.2, -3.6) {$-$};
\node[op] (s2) at (1.2, -4.2) {$-$};
\node[op] (s3) at (1.2, -4.8) {$-$};
\node[annot, below=0.0cm of s3] {$\boldsymbol{\mu}_x$};

% Residual weights -> sum
\node[neuron, minimum size=0.7cm, fill=red!8] (rsum) at (3.2, -4.2) {$\Sigma$};
\node[annot, below=0.15cm of rsum, text=red!60!black] {residual};

% Mean scalar
\node[neuron, minimum size=0.7cm, fill=blue!8] (mk) at (3.2, -2.7) {$m_k$};
\node[annot, above=0.05cm of mk, text=blue!60!black] {mean};

% Combine
\node[op] (plus) at (4.6, -3.5) {$+$};
\node[op] (ract) at (5.6, -3.5) {$\sigma$};
\node[neuron] (rak) at (6.7, -3.5) {$a_k$};

% Input -> centering
\draw[->] (rx1) -- ++(0.4, 0) |- (s1);
\draw[->] (rx2) -- ++(0.3, 0) |- (s2);
\draw[->] (rx3) -- ++(0.2, 0) |- (s3);

% Input -> mean path
\draw[->, blue!40, thick] (rx1) -- ++(0.4, 0) |- (mk);

% Centering -> residual weights
\draw[->, red!50, thick] (s1) -- node[above, annot, pos=0.6] {$v_1$} (rsum);
\draw[->, red!50, thick] (s2) -- node[above, annot, pos=0.6] {$v_2$} (rsum);
\draw[->, red!50, thick] (s3) -- node[above, annot, pos=0.55] {$v_3$} (rsum);

% Residual + Mean -> combine
\draw[->, red!50, thick] (rsum) -- (rsum -| plus.south) -- (plus);
\draw[->, blue!40, thick] (mk) -- (mk -| plus.north) -- (plus);
\draw[->] (plus) -- node[above, annot] {$z_k$} (ract);
\draw[->] (ract) -- (rak);

% Brace for regularization
\draw[decorate, decoration={brace, amplitude=4pt, mirror}, red!60!black]
    (rsum.south west) ++(0, -0.55) -- node[below=3pt, font=\scriptsize, text=red!60!black] {$\|\mathbf{v}_k\|_2 \to 0$} ++(1.8, 0);

\end{tikzpicture}
\end{center}

\vspace{0.1cm}

\begin{center}
\begin{tabular}{lcc}
\toprule
Regularization target & Effect on $\mathbb{E}[z_k]$ & Effect on $\text{Var}(z_k)$ \\
\midrule
Weight decay on $\mathbf{w}_k$ & $\to 0$ (destroyed) & $\to 0$ \\
\textbf{Residual-only on} $\mathbf{v}_k$ & Preserved ($= m_k$) & $\to 0$ \\
\bottomrule
\end{tabular}
\end{center}

# Training

**$\boldsymbol{\mu}_x$ estimation.** Computed from a calibration forward pass (same infra as VBP stats).
**ViT:** collect $\mathbb{E}[\text{input to fc1}]$ per layer. **CNN+BN:** $\boldsymbol{\mu}_x \approx \boldsymbol{\beta}$ (BN bias), free.

**$\boldsymbol{\mu}_x$ refresh** (optional, every $N$ PAT epochs; $N{=}\texttt{None}$ to never refresh).
When the model changes during PAT, $\mathbb{E}[\mathbf{x}]$ drifts from the frozen $\boldsymbol{\mu}_x$.
Refresh re-estimates $\boldsymbol{\mu}_x$ and adjusts $m_k$ to preserve the function exactly:

$$\boldsymbol{\mu}_x' \leftarrow \mathbb{E}_\text{calib}[\mathbf{x}], \qquad m_k \leftarrow m_k + \mathbf{v}_k^\top(\boldsymbol{\mu}_x - \boldsymbol{\mu}_x'), \qquad \boldsymbol{\mu}_x \leftarrow \boldsymbol{\mu}_x'$$

Without refresh, the system still converges (regularizing $\mathbf{v}_k \to 0$ kills both variance and any leaked mean signal), but refresh keeps the decomposition clean so $\lambda$ can be smaller.

**Training loss** with column-wise $L_{2,1}$ group regularization on residual weights only:

$$\mathcal{L} = \mathcal{L}_\text{task}(\theta) \;+\; \lambda \underbrace{\sum_{l} \sum_{k=1}^{n_\text{in}^{(l)}} \left\| V^{(l)}_{:,k} \right\|_2}_{\text{sum over input channels (intermediate dim) in all reparam layers}}$$

Here $k$ indexes the **input channels** of the reparam layer (= intermediate dimension for fc2). When $\|V_{:,k}\| \to 0$, fc2 ignores variation in intermediate channel $k$ --- its mean contribution $\mathbb{E}[a_k] \cdot W_{:,k}$ is already absorbed into $m$.

Note: standard weight decay is **disabled** for $m_k$ and $\mathbf{v}_k$; other parameters keep normal weight decay.

**Gradient** (column-wise subgradient of group lasso): $\;\dfrac{\partial \mathcal{L}_\text{reg}}{\partial V_{:,k}} = \lambda \cdot \dfrac{V_{:,k}}{\|V_{:,k}\|_2}$. Pushes each column $V_{:,k}$ radially toward zero as a group --- all output neurons' dependence on input channel $k$ shrinks together.

# Pruning

When $\|\mathbf{v}_k\| \approx 0$ after training:

1. Channel output $\approx \sigma(m_k)$ --- a learned constant
2. Fold: $\;\mathbf{b}_{l+1} \mathrel{+}= W_{l+1}[:,k] \cdot \sigma(m_k)$
3. Remove channel $k$

This is the VBP compensation formula, but now *accurate* --- variance is genuinely near-zero.

We regularize fc2's **input columns** ($V_{:,k}$), targeting fc2's input channels = the intermediate dimension that VBP prunes via fc1. This is input-side targeting on fc2, self-contained per MLP block and aligned with skip connections (the residual stream dimension $d_\text{model}$ is never pruned).
