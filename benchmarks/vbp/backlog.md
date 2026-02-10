# VBP Research Backlog

## 1. VBP for Standard CNNs

**Idea:** Extend VBP to standard convolutional architectures (ResNet, MobileNet, EfficientNet) beyond the current ViT and ConvNeXt support.

**Motivation:** The VBP paper focuses on ViT and ConvNeXt, both of which have clear MLP bottleneck structures (fc1→GELU→fc2 or pwconv1→GELU→pwconv2). Standard CNNs have a different topology — Conv→BN→ReLU blocks, residual additions, depthwise-separable convolutions — and it's unclear if VBP's post-activation variance importance transfers directly. However, the core insight (variance of post-nonlinearity activations correlates with channel importance) should generalize.

**Architecture-specific considerations:**

- **ResNet:** Each residual block has two Conv3x3 layers (or 1x1→3x3→1x1 bottleneck). The activation function is ReLU (not GELU). VBP stats should be collected after ReLU. For bottleneck blocks, the intermediate 3x3 conv is analogous to the MLP intermediate dim — MLP-only pruning would target only this dimension, preserving the residual stream width. The skip connection constraint means the last conv in each block (output to residual stream) must stay unpruned, matching the ViT/ConvNeXt pattern.
- **MobileNet / EfficientNet:** Inverted residual blocks with depthwise separable convolutions: expand (1x1) → depthwise (3x3) → squeeze (1x1). The expand conv is analogous to fc1/pwconv1 — it's the intermediate dimension. MLP-only pruning would target the expansion factor. Depthwise conv channels are tied 1:1 to the expanded dim, so pruning the expand conv automatically prunes the depthwise.
- **BatchNorm interaction:** Unlike LayerNorm in ViT, BN normalizes per-channel across the batch spatial dimensions. Post-BN activations have unit variance by construction within a batch — so stats must be collected *after BN + ReLU*, not after BN alone. The ReLU breaks the unit-variance property and reintroduces channel-discriminative variance.
- **Tensor format:** Standard CNNs use NCHW throughout (unlike ConvNeXt's NHWC Linear layers), so no permutation needed in the stats hook. Simpler than the ConvNeXt case.

**Implementation plan:**
1. Add ResNet support to `vbp_imagenet.py` (model loading, ignored_layers, target_layers)
2. For bottleneck ResNets: MLP-only = prune only the intermediate 3x3 conv dimension
3. For basic-block ResNets: both convs in a block share the same output dim, so pruning is per-block
4. Validate on ResNet-50 with ImageNet — compare retention against magnitude pruning and Taylor importance
5. Extend to MobileNetV2 / EfficientNet-B0

**Key questions:**
- Does VBP's variance criterion outperform magnitude/Taylor pruning for CNNs, or is the advantage specific to Transformer MLP structures?
- How does the bias compensation step perform with BN layers? BN has both scale (gamma) and shift (beta) parameters that interact with the mean-shift correction.
- For ResNet skip connections with dimension mismatch (downsampling blocks), should the 1x1 projection conv be pruned or ignored?

**Expected outcome:** A unified VBP pipeline in Torch-Pruning that works across ViT, ConvNeXt, ResNet, and MobileNet, establishing VBP as a general-purpose importance criterion rather than a Transformer-specific technique.

## 2. Debug ConvNeXt paper vs. our implementation

**Problem:** The VBP paper reports 81.3% top-1 on ConvNeXt-T after pruning + fine-tuning, but our implementation achieves only 80.9% — a 0.4% gap.

**Investigation so far:** Running with the exact author hyperparameters (lr, batch size, schedule) gives a worse result of ~-1% vs. paper. Adjusting lr and batch size to values better suited to our setup recovers most of the gap, reaching -0.4%. The remaining difference may come from fine-tuning schedule details (warmup, cosine decay length, EMA), data augmentation differences (RandAugment/Mixup/CutMix settings), or subtle differences in how bias compensation interacts with the BN/LN statistics after pruning.

**Next steps:**
- Compare fine-tuning configs line-by-line against the paper's released code (if available)
- Ablate: does the gap come from the pruning step (retention accuracy) or the fine-tuning step?
- Check if EMA or stochastic depth settings differ