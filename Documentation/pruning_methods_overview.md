# Pruning Methods Overview

Pruning is the process of removing parameters (either individually or in groups, such as by neurons) from an existing network. The goal is to maintain the network’s accuracy while increasing its efficiency. Pruning is typically divided into three types:

- **Unstructured Pruning:** Removes individual parameters (weights).
- **Semi-Structured (Pattern-Based) Pruning:** Removes parameters in specific patterns (requires special hardware/software support).
- **Structured Pruning:** Removes entire groups (e.g., channels or filters) and is the only approach that achieves universal acceleration and compression without special support.

Our framework supports:

- **Channel Pruning** as structured pruning.
- **Slice Pruning** as semi-structured (pattern-based) pruning.

---

## Channel Pruning

Channel pruning focuses on removing entire channels (filters) from convolutional layers to generate a reduced, more efficient architecture.

**Mathematical Formulation:**

Consider a convolutional layer represented by a 4D tensor with shape $`C_{out} \times C_{in} \times H \times W`$, where:
- $`C_{out} `$ is the number of filters,
- Each filter is a 3D tensor containing $`C_{in}`$ kernels,
- Each kernel is a 2D weight matrix of size $`H \times W`$ (typically $`3 \times 3`$).

By pruning output channels (i.e., removing entire filters), the output of the layer is reduced from $`C_{out}`$ to a lower number, thereby simplifying the subsequent layers.

**Training Note:**  
During training, channels are not physically removed but are instead masked (set to zero). This strategy facilitates checkpoint saving/loading and provides finer control over the pruning rate. When the target pruning rate or MAC (Multiply-Accumulate Operations) reduction is reached, the model is physically pruned to produce a reduced graph.

> **Checkpoint Loading Note:**  
> When loading a pruned checkpoint with the original code, shape mismatches may occur. To address this, our framework extends the standard loading procedure:
>
> ```python
> try:
>     model.load_state_dict(weights)
> except Exception:
>     from torch_pruning.utils import load_state_dict_pruned
>     model = load_state_dict_pruned(model, weights)
> ```

---

## Slice Pruning

Slice pruning operates at a finer granularity by removing “slices” of filters based on specific patterns defined by the SNP team. This method is semi-structured and typically requires additional hardware or software support.

**Mathematical Formulation:**

Consider the same convolutional layer with shape: $`C_{out} \times C_{in} \times H \times W`$,

For slice pruning, we proceed as follows:

1. **Block Division:**

   The $`C_{out}`$ filters are divided into blocks. For example, if each block contains $`A`$ filters (e.g., $`A = 8`$), then the number of blocks $`B`$ is given by

   $$
   B = \frac{C_{out}}{A}
   $$

2. **Slice Extraction:**

   Within each block, the weights are reorganized so that each block can be viewed as consisting of multiple 1D slices. The total number of slices $`S`$ in the layer is then:

   $$
   S = B \times C_{in} \times H \times W
   $$

3. **Importance Evaluation & Pruning:**

   An importance metric (typically based on the $`L_2`$-norm) is computed for each slice. Slices with low importance are pruned (set to zero), which reduces latency and optimizes the network further.

**Additional SNP Constraints:**

- The first index of each row in a kernel must remain unpruned.
- In our target hardware (Thetis), the maximum sequential zero-skip in a kernel line is up to 4 pixels, which is acceptable for typical $`3 \times Y`$ kernels.
- The ordering of filters for block formation may vary based on hardware clusters. For instance, with two clusters and $`C_{out}`$ filters, one possible ordering is:
  
  $$
  [0, B, 2B, \ldots, (A-1)B,\; 1, B+1, 2B+1, \ldots, (A-1)B+1,\; \ldots,\; B-1, 2B-1, \ldots, AB-1]
  $$
  
  In general, this ordering can be expressed in Python as:
  
  ```python
  concat([[b + a for b in range(0, A * B, B)] for a in range(B)])
  ```
  
- For depth-wise convolutions (where each input channel is convolved with its own filter), the weight tensor is typically of shape
  
  $$
  C_{out} \times 1 \times H \times W
  $$
  
  and the block size $`A`$ might differ (e.g., $`A = 2`$).

---

## Summary

- **Channel Pruning:**  
  Removes entire filters (channels) from a layer. This leads to significant model size reduction and computational efficiency without requiring special hardware.

- **Slice Pruning:**  
  Provides a more granular approach by pruning slices within filters. This method complements channel pruning by further optimizing the network, though it generally requires specialized hardware/software support.

Together, these pruning techniques enable flexible and hardware-aware network optimization.
