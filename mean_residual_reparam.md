---
title: "Variance-Normalized Reparameterization (VNR)"
geometry: margin=2.5cm
fontsize: 11pt
header-includes:
  - \usepackage{amsmath,amssymb}
  - \usepackage{bm}
  - \let\boldsymbol\bm
  - \usepackage{booktabs}
  - \usepackage{enumitem}
  - \usepackage{tikz}
  - \usetikzlibrary{positioning,arrows.meta,calc,fit,backgrounds,decorations.pathreplacing}
  - \setlist{nosep}
---

# Problem

VBP measures activation variance but has no training-time lever to push it down. Plain weight decay (WD) destroys both mean and variance. Mean-residual reparameterization ($V = W$, regularize $V$) separates mean from variance, but is **scale-blind**: all channels get equal penalty regardless of input scale $\sigma_{x,k}$. The network can **compensate** by inflating $\sigma$ upstream (e.g.\ via $\gamma$ in a preceding BN) while shrinking $V$, preserving variance without penalty.

# VNR: Input Normalization via BN(affine=False)

Inserts $\text{BN}(\texttt{affine=False})$ before the target layer. The trainable weight $\tilde{V} = W \cdot \sigma_\text{cal}$ operates on normalized input:

$$\boxed{\mathbf{x}_\text{bn} = \text{BN}(\mathbf{x}), \qquad z_k = \tilde{V}_k^\top \mathbf{x}_\text{bn} + m_k} \qquad \text{(function-preserving at init)}$$

Initialization: $\tilde{V} = W \cdot \sigma_\text{cal}$, $m_k = b_k + \mathbf{w}_k^\top \boldsymbol{\mu}_\text{cal}$, BN running stats $= (\boldsymbol{\mu}_\text{cal}, \sigma_\text{cal}^2)$. Weight magnitude **directly indicates importance**: $\|\tilde{V}_{:,k}\| = \sigma_k \cdot \|W_{:,k}\|$ measures the variance contribution of channel $k$. BN auto-updates running stats during training --- no frozen $\sigma$, no drift risk.

\vspace{0.3cm}

\begin{center}
\begin{tikzpicture}[
    >=Stealth,
    neuron/.style={draw, circle, minimum size=0.55cm, inner sep=0pt, font=\scriptsize},
    op/.style={draw, circle, inner sep=1.5pt, font=\scriptsize},
    annot/.style={font=\scriptsize, text=black!60},
    block/.style={draw, rounded corners, minimum height=0.5cm, inner sep=3pt, font=\scriptsize},
]

% === Standard (top) ===
\node[annot, font=\small\bfseries] at (-1.8, 0) {Standard:};

\node[neuron] (x1) at (0, 0.8) {$x_1$};
\node[neuron] (x2) at (0, 0) {$x_2$};
\node[neuron] (x3) at (0, -0.8) {$x_3$};
\node[neuron, minimum size=0.7cm] (z) at (2.5, 0) {$\Sigma$};
\node[annot, above=0.05cm of z] {$b_k$};
\node[op] (act) at (3.7, 0) {$\phi$};
\node[neuron] (ak) at (4.9, 0) {$a_k$};

\draw[->, thick] (x1) -- node[above, annot, pos=0.4] {$w_1$} (z);
\draw[->, thick] (x2) -- node[above, annot, pos=0.5] {$w_2$} (z);
\draw[->, thick] (x3) -- node[below, annot, pos=0.4] {$w_3$} (z);
\draw[->] (z) -- node[above, annot] {$z_k$} (act);
\draw[->] (act) -- (ak);

% === VNR (bottom) ===
\node[annot, font=\small\bfseries] at (-1.8, -3.8) {VNR:};

\node[neuron] (rx1) at (0, -2.7) {$x_1$};
\node[neuron] (rx2) at (0, -3.5) {$x_2$};
\node[neuron] (rx3) at (0, -4.3) {$x_3$};

% BN normalization nodes
\node[block, fill=green!8] (n1) at (1.4, -2.7) {BN};
\node[block, fill=green!8] (n2) at (1.4, -3.5) {BN};
\node[block, fill=green!8] (n3) at (1.4, -4.3) {BN};

% Residual weights -> sum
\node[neuron, minimum size=0.7cm, fill=red!8] (rsum) at (3.8, -3.8) {$\Sigma$};

% Mean scalar
\node[neuron, minimum size=0.7cm, fill=blue!8] (mk) at (3.8, -2.5) {$m_k$};
\node[annot, above=0.05cm of mk, text=blue!60!black] {trainable, WD=0};

% Combine
\node[op] (plus) at (5.1, -3.2) {$+$};
\node[op] (ract) at (6.1, -3.2) {$\phi$};
\node[neuron] (rak) at (7.2, -3.2) {$a_k$};

% Input -> normalize
\draw[->] (rx1) -- (n1);
\draw[->] (rx2) -- (n2);
\draw[->] (rx3) -- (n3);

% Normalize -> residual weights
\draw[->, red!50, thick] (n1) -- node[above, annot, pos=0.55] {$\tilde{v}_1$} (rsum);
\draw[->, red!50, thick] (n2) -- node[above, annot, pos=0.6] {$\tilde{v}_2$} (rsum);
\draw[->, red!50, thick] (n3) -- node[above, annot, pos=0.55] {$\tilde{v}_3$} (rsum);

% Residual + Mean -> combine
\draw[->, red!50, thick] (rsum) -- (rsum -| plus.south) -- (plus);
\draw[->, blue!40, thick] (mk) -- (mk -| plus.north) -- (plus);
\draw[->] (plus) -- node[above, annot] {$z_k$} (ract);
\draw[->] (ract) -- (rak);

% Brace for regularization -- horizontal, below sum node
\draw[decorate, decoration={brace, amplitude=4pt, mirror}, red!60!black]
    (2.8, -4.7) -- node[below=3pt, font=\scriptsize, text=red!60!black] {$\|\tilde{V}_k\| = \sigma_k \|W_k\| \to 0$} (5.0, -4.7);

% Compensation-blocked annotation -- below brace
\node[annot, text=green!50!black, font=\scriptsize\itshape] at (1.4, -5.5) {BN(affine=False) blocks $\sigma$ compensation};

\end{tikzpicture}
\end{center}

\vspace{0.1cm}

\begin{center}
\begin{tabular}{lcc}
\toprule
& Plain ($V{=}W$) & VNR ($\tilde{V}{=}W \cdot \sigma$) \\
\midrule
Reg penalty on ch $k$ & $\propto \|W_{:,k}\|$ (scale-blind) & $\propto \sigma_k \|W_{:,k}\|$ (= var contrib) \\
Compensation via $\uparrow\sigma$ & possible & blocked ($\|\tilde{V}\|$ grows $\Rightarrow$ more penalty) \\
Weight magnitude & weight size only & true importance \\
\bottomrule
\end{tabular}
\end{center}

# Training

**Loss** with column-wise $L_{2,1}$ on normalized weights ($k$ = input channel = intermediate dim):

$$\mathcal{L} = \mathcal{L}_\text{task}(\theta) \;+\; \lambda \sum_{l} \sum_{k} \left\| \tilde{V}^{(l)}_{:,k} \right\|_2$$

Standard WD is **disabled** for $m_k$ and $\tilde{V}$; other parameters keep normal WD.

**Statistics update**: BN(affine=False) auto-updates running mean and variance during training (exponential moving average, momentum=0.1). No manual `refresh_stats` needed --- the normalization is exact per-batch, eliminating $\sigma$ drift.

# Pruning

When $\|\tilde{V}_{:,k}\| \approx 0$: channel output $\approx \phi(m_k)$ (learned constant). Fold into next layer bias, remove channel. This is the VBP compensation formula, now accurate since variance is genuinely near-zero.

**Importance criterion.** After merge-back, $W_\text{eff} = \tilde{V} / \sigma_\text{BN}$, so $\|W_{\text{eff},:,k}\| \cdot \sigma_k = \|\tilde{V}_{:,k}\|$. The \texttt{weight\_variance} importance mode ($I_k = \|W_{\text{fc2},:,k}\|_2 \cdot \sigma_k$) is therefore the natural pruning criterion for VNR: it equals the $\tilde{V}$ column norm that regularization explicitly minimized.

**Related:** BN $\gamma$ pruning (Liu 2017 / Ye 2018) targets *output* channels where nonlinearities create compensation paths; our input-side normalization avoids this. Network whitening (Luo 2017; Kang \& Park, NeurIPS 2024) decorrelates via $\Sigma^{-1/2}$; our similarity discount in `VarianceImportance` is a softer version.
