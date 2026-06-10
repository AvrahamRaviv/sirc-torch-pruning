# Archived diagnostics

One-off verification scripts from the normnet propagation work (May–Jun 2026). Kept for reference; superseded by `normnet_main.py`.

- `normnet_standalone.py` — single-device normnet pipeline (pre-DDP). `sanity_propagation.py` imports it.
- `sanity_propagation.py` — asserts code propagation matches the PDF form (run: `python sanity_propagation.py` → PASS/FAIL).
- `mlp_prop_check.py` — residual-MLP propagation criterion vs brute-force drop-one.
- `mlp_mnist_check.py` — MNIST sandbox for the NCI input-independence assumption (motivated `nci_cov`).
- `toy_nonrel_doublecount.py` — algebraic check that non-relative propagation doesn't double-count σ.
- `generate_channel_dumps_experiments.py` — experiment generator for `normnet_standalone.py` channel dumps.
