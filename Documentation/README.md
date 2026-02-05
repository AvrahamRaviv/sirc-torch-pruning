# Documentation

This folder contains comprehensive explanations, examples, and integration guidelines for the SIRC Torch Pruning framework. It covers usage instructions, underlying methods, and practical examples.

## Contents

- **Pruning Methods Overview:**  
  Detailed description of channel and slice pruning approaches is available in [pruning_methods_overview.md](pruning_methods_overview.md).

- **Usage & Integration:**  
  Step-by-step instructions on initializing and using the pruning functionality, including:
  - **Initialization:** Creating the Pruning object.
  - **Regularization:** Incorporating channel and slice regularization during training.
  - **Pruning Execution:** How and when to prune (masking during training vs. physical pruning during checkpoint saving).

## Key Functions

### Initialization
Instantiate the `Pruning` class:
```python
from torch_pruning.utils import Pruning

model = ...  # your model definition
forward_fn = model.forward  # or a custom forward function
pruner = Pruning(model, config_folder, forward_fn)
```

### Regularization

- **Channel Regularization:**  
  After `loss.backward()`, apply:
  ```python
  pruner.channel_regularize(model)
  ```

- **Slice Regularization:**  
  During the loss computation, include:
  ```python
  loss += pruner.slice_regularize(model)
  ```

### Pruning
At a given epoch:
```python
pruner.prune(model, current_epoch, log=my_logger)
```
*Note:* During training, channels are masked (set to zero) rather than physically removed. This ensures easier checkpoint management. When the target pruning rate or MAC reduction is reached, a physical pruning step generates a reduced model.

### Handling Checkpoints
To load pruned checkpoints without encountering shape mismatches, use:
```python
try:
    model.load_state_dict(weights)
except Exception:
    model, missing_keys, unexpected_keys = torch_pruning.load_state_dict_pruned(model, weights)
```

## Hyperparameter Tuning for Pruning

### Overview of Pruning Workflow

The pruning process is divided into three main stages:

1. **Regularization Phase**  
   - Encourage sparsity by reducing the L2 norm of coupled channels.  
   - Train with different regularization coefficients until enough channels are suppressed.  

2. **Pruning Phase**  
   - Define pruning steps (start and end epochs) with a MAC target.  
   - Mask channels with zeros during training and physically remove them once the target is reached.  

3. **Fine-Tuning Phase**  
   - Fine-tune the reduced architecture to recover task performance.  
   - Handle checkpoint saving/loading with special utilities to avoid shape mismatches.  


Pruning during training requires careful selection of several hyperparameters, including:

- Initial learning rate and its scheduler  
- Total number of training epochs  
- Number of epochs per stage  
- Regularization strength and pruning schedule  
- And more  

This section provides practical guidance based on our experience for handling each stage and debugging results.

### Step 1: Regularization Phase

The goal of this phase is to reduce the L2 norm of coupled channels as preparation for pruning.  
We use [sparse training mode](https://github.com/VainF/Torch-Pruning/tree/master?tab=readme-ov-file#sparse-training-optional), which requires choosing a regularization coefficient.

**Recommendation:** run three experiments with the following regularization strengths: reg = [5e-2, 5e-3, 5e-4]

After sufficient training epochs, you should observe a drop in $\ell_2$ norms with minimal accuracy degradation.

To analyze the regularization process, use the provided `analyze.py` script.  
This script scans through training checkpoints and plots L2 norm histograms across epochs during the regularization phase.

When you identify a checkpoint where a sufficient number of channels have low amplitude, you can use that epoch as the starting point for the pruning phase.

### Step 2: Pruning Phase

The ideal behavior is to set a **MAC target** (e.g., 70%, which is equivalent to a 30% MAC reduction), and then run gradual pruning until the target is reached.  

To achieve this, we define **pruning steps** by specifying the start and end epochs. To make pruning less harsh, the schedule is calculated as if the target were 100%, and pruning automatically stops once the desired target is reached.  

For stability reasons, during this phase we do not immediately remove channels. Instead, channels are masked with zeros. Once the target is reached, we physically prune the masked channels, resulting in a reduced architecture.  

### Step 3: Fine-Tuning Phase

Once the reduced architecture is obtained, we fine-tune it as usual, aiming to recover the original task performance.  

After pruning, many channels are zeroed out, which drastically changes the distribution of weights.  
We have found that **resetting the optimizer** is very useful, since it clears biased optimizer statistics and stabilizes training.  

There are two ways to reset:  
1. **Restart the job** (simple but less convenient).  
2. **Use the built-in mechanism** we provide, which stores the reset flag internally and requires only a few extra lines after calling `pruner.prune`:  

```python
if pruner.channel_pruner.reset_optimizer:
    optimizer = create_optimizer()  # EDIT THIS LINE: replace with your optimizer initialization
    pruner.channel_pruner.reset_optimizer = False
```

This ensures fine-tuning starts with a “clean” optimizer state, while keeping the pruned architecture intact.

> ⚠️ **Note on saving and loading:**  

Since the graph has been pruned, saving and reloading checkpoints (e.g., for preemption) may cause mismatches. This happens because the model graph is initialized as the original architecture, while the pruned weights belong to the reduced one.  

To address this, we implemented a wrapper around `torch.load`.  
To load pruned checkpoints without encountering shape mismatches, use:

```python
try:
    model.load_state_dict(weights)
except Exception:
    model, missing_keys, unexpected_keys = torch_pruning.load_state_dict_pruned(model, weights)
```

### Layer Selection

Choosing which layers to prune is controlled by the `"layers"` entry in the config.  
This entry should contain a list of `Conv2d` layers to prune.  

To easily list all `Conv2d` layers in your model, run the following in debug mode:

```python
for name, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        print(name)
```

> ⚠️ Note: `ConvTranspose` layers are currently not supported.

#### Recommendation

For your first run, specify **a single layer** to prune. This helps verify that the mechanism works as expected before scaling up.

We provide detailed logs about:

* Which layers are selected for pruning
* Their dependent layers (i.e., layers that must be updated consistently)

**Example log output:**

```
*************
Group number 1:
Source conv: layer1.conv1
Dependencies:
layer1.conv2
layer1.conv3
*************

There are 3 groups of layers, with the following source convs:
['layer1.conv1', 'layer2.conv1', 'layer3.conv1']
```

* **Group number**: an index of the group.
* **Source conv**: the main convolution selected for pruning.
* **Dependencies**: other conv layers connected to it that are pruned together.
* **Summary line**: total groups and their source conv names.

## Configuration

Save your pruning configuration as `pruning_config.json` in the output folder. Below is an example configuration:

### Channel Pruning Parameters (`channel_sparsity_args`)

- **is_prune** *(bool)*: Enable or disable channel pruning.
- **global_pruning** *(bool)*: Use global_pruning algorithm.
- **block_size** *(int)*: Defines the grouping size ("round_to") during pruning.
- **start_epoch** *(int)*: The epoch when channel pruning begins.
- **end_epoch** *(int)*: The epoch when channel pruning stops.
- **epoch_rate** *(int)*: Frequency (in epochs) at which pruning is executed.
- **global_prune_rate** *(float)*: Overall fraction of channels to prune.
- **max_pruning_rate** *(float)*: Maximum allowed fraction of channels per layer to be pruned.
- **mac_target** *(float)*: Target reduction factor for MACs.
- **reach_mac_target** *(bool)*: Indicates if reduction target already reached.
- **log_str** *(string)*: If reach_mac_target is true, log_str contains pruning information.
- **pruning_method** *(string)*: The algorithm used (e.g., "GroupNormPruner").
- **prune_channels_at_init** *(bool)*: If true, channels are pruned at model initialization.
- **infer** *(bool)*: Indicates if the model is in inference mode.
- **isomorphic** *(bool)*: Use isomorphic idea during pruning if enabled.
- **regularize** *(object)*: Contains regularization settings:
  - **reg** *(float)*: Base regularization strength.
  - **mac_reg** *(float)*: Additional regularization specific to MAC reduction.
  - **gamma** *(float)*: Scaling factor for the regularization term.
- **MAC_params** *(object)*: Settings for MAC-based weighting (relevant for MACAwareImportance):
  - **use_macs** *(bool)*: Enable MAC-based adjustments.
  - **type** *(string)*: Aggregation method (e.g., "Sum").
  - **alpha** *(float)*: Weight factor for MAC calculations.
  - **beta** *(int)*: Secondary weight factor for MAC calculations.
- **layers** *(list)*: Specifies which Conv2d layers to prune.
- **input_shape** *(list or tuple)*: either a single shape `[B, C, H, W]` or a list of shapes `[[...], [...], ...]`.  
- **container** *(string, optional)*: default is `None`. When "tuple" is set, the returned tensors are packed into a tuple instead of the default list.

### Slice Pruning Parameters (`slice_sparsity_args`)

- **is_prune** *(bool)*: Enable or disable slice pruning.
- **block_size** *(int)*: The size of the block used for grouping filters into slices.
- **start_epoch** *(int)*: The epoch when slice pruning starts.
- **end_epoch** *(int)*: The epoch when slice pruning ends.
- **epoch_rate** *(int)*: Frequency (in epochs) for applying slice pruning.
- **prune_rate** *(float)*: The fraction of slices to prune.
- **pruning_gradually** *(bool)*: If true, the pruning is applied gradually over epochs.
- **pruning_mode** *(string)*: Mode of operation.
- **disable_first_index** *(bool)*: If true, the first index in each kernel is preserved.
- **pruning_method** *(string)*: The metric or algorithm used for slice pruning (e.g., "L2_norm").
- **reg** *(float)*: Regularization strength applied during slice pruning.

This configuration file allows you to customize the pruning process to match your model's requirements and hardware constraints. Adjust these parameters to balance efficiency and model performance.

## Future Enhancements

- **Hardware-Aware Algorithms:**  
  Integration of MAC-aware and latency-aware (SNP-based) pruning methods.

- **Transformer Support:**  
  Upcoming support for pruning attention mechanisms, including token and multi-head attention components.

For complete integration examples, refer to the [examples](examples) folder.
