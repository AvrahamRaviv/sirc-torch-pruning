import os
import json
from enum import Enum
from functools import partial
from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
import torch_pruning as tp

try:
    import spyq  # noqa: F401
except Exception:
    spyq = None

from torch_pruning.utils import count_ops_and_params

"""
SIRC Torch-Pruning, last update: 02/02/2026
Owners: Avraham Raviv, Ishay Goldin.
"""


def _log(log, msg: str) -> None:
    """Dispatch a message to a logger or stdout."""
    if log is not None:
        log.info(msg)
    else:
        print(msg)


_IMAGE_KEYS = ("image", "images", "img", "img1", "input", "inputs",
               "pixel_values", "x", "data")


def _unpack_images(batch):
    """Extract image tensor from a dataloader batch.

    Handles three formats:
    - Tensor: returned as-is
    - tuple/list: returns first element (standard (images, labels) convention)
    - dict: looks up common image keys, falls back to first Tensor value
    """
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, dict):
        for key in _IMAGE_KEYS:
            if key in batch:
                return batch[key]
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v
    raise ValueError(f"Cannot extract images from batch of type {type(batch)}. "
                     f"Expected Tensor, tuple/list, or dict with one of {_IMAGE_KEYS}.")


def _recalibrate_bn(model, train_loader, device, log=None):
    """Recalibrate BN running stats after structural pruning."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
    model.train()
    total = min(100, len(train_loader))
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 100:
                break
            images = _unpack_images(batch)
            model(images.to(device, non_blocking=True))
    model.eval()
    _log(log, f"BN recalibration done ({total} batches)")


class PruningMethod(str, Enum):
    """Supported channel-pruning methods (str mixin for JSON compat)."""
    BN_SCALE = "BNScalePruner"
    GROUP_NORM = "GroupNormPruner"
    MAC_AWARE = "MACAwareImportance"
    VBP = "VBP"
    LAMP = "LAMP"
    RANDOM = "Random"


class Pruning:
    """
    High-level pruning interface combining channel and slice pruning.
    """
    def __init__(self, model: nn.Module, config_folder: str, forward_fn: Optional[Any] = None,
                 log: Optional[Any] = None, device: Optional[torch.device] = None,
                 train_loader=None) -> None:
        self.version = "1.1.5"
        try:
            config_path = os.path.join(config_folder, "pruning_config.json")
            with open(config_path, "r") as f:
                sparsity_args = json.load(f)
            channel_sa = sparsity_args['channel_sparsity_args']
            slice_sa = sparsity_args['slice_sparsity_args']
            _log(log, "=> Init pruner module")
        except Exception as e:
            if os.path.exists(os.path.join(config_folder, "pruning_config.json")):
                print("There is pruning_config.json in output folder, but it is failed with the following error:")
                print(e)
            else:
                print("There is no pruning_config.json in output folder")
            channel_sa, slice_sa = None, None
            _log(log, "=> Unable to find a valid pruning configuration.")

        self.channel_pruner = ChannelPruning(channel_sa, model, config_folder, forward_fn, log, device,
                                             train_loader=train_loader)
        self.slice_pruner = SlicePruning(slice_sa, model, log)
        # Synchronize slice block size and channel mask dictionary between pruners.
        self.channel_pruner.slice_block_size = self.slice_pruner.block_size
        self.slice_pruner.channel_mask_dict = self.channel_pruner.channel_mask_dict

    def channel_regularize(self, model: nn.Module) -> float:
        """Apply channel regularization to the model."""
        return self.channel_pruner.regularize(model)

    def slice_regularize(self, model: nn.Module) -> torch.Tensor:
        """Apply slice regularization to the model."""
        return self.slice_pruner.regularize(model)

    def prune(self, model: nn.Module, epoch: int, log: Optional[Any] = None,
              mask_only: bool = True, train_loader=None) -> None:
        """Perform pruning using both channel and slice pruners."""
        self.channel_pruner.prune(model, epoch, log=log, mask_only=mask_only,
                                  train_loader=train_loader)
        self.slice_pruner.prune(model, epoch, log=log)


class ChannelPruning:
    """
    Implements channel pruning using a chosen pruning method.
    """
    def __init__(self, channel_sparsity_args: Optional[Dict[str, Any]], model: nn.Module,
                 config_folder: str, forward_fn: Any, log: Optional[Any] = None,
                 device: Optional[torch.device] = None, train_loader=None) -> None:
        self.channel_mask_dict: Dict[str, torch.Tensor] = {}
        self.log = log
        if channel_sparsity_args is None:
            _log(self.log, "=> Unable to find a valid channel pruning configuration.")
            self.prune_channels = False
            self.prune_channels_at_init = False
            return

        self.prune_channels = channel_sparsity_args.get('is_prune', True)
        if not self.prune_channels:
            _log(self.log, "=> Channel pruning already complete (is_prune=False), skipping init.")
            self.prune_channels_at_init = False
            self.slice_block_size = None
            return

        self.channel_sparsity_args = channel_sparsity_args
        self.device = device
        self.infer = self.channel_sparsity_args['infer']

        # Example inputs for forward pass estimation.
        self.example_inputs = build_inputs(channel_sparsity_args, device=device)
        if self.infer:
            self.example_inputs = (self.example_inputs[0][:, :1], self.example_inputs[0][:, :1])
        self.current_epoch: int = 0
        self.ignored_layers: List[Any] = []
        self.start_epoch: int = self.channel_sparsity_args['start_epoch']
        self.end_epoch: int = self.channel_sparsity_args['end_epoch']
        self.epoch_rate: int = self.channel_sparsity_args['epoch_rate']
        self.global_prune_rate: float = self.channel_sparsity_args.get('global_prune_rate', 0.0)
        self.layers_to_prune = self.channel_sparsity_args['layers']
        self.mac_target: float = self.channel_sparsity_args.get('mac_target', 0.0)
        self.prune_channels_at_init = self.channel_sparsity_args['prune_channels_at_init'] or self.infer

        self.pruning_method = PruningMethod(self.channel_sparsity_args['pruning_method'])
        self.channels_pruner_args = {
            "pruning_method": self.pruning_method,
            "global_pruning": self.channel_sparsity_args['global_pruning'],
            "round_to": self.channel_sparsity_args['block_size'],
            "reg": self.channel_sparsity_args['regularize']['reg'],
            "mac_reg": self.channel_sparsity_args['regularize']['mac_reg'],
            "gamma_reg":  self.channel_sparsity_args['regularize'].get("gamma", 1),
            "alpha_shrinkage_reg":  self.channel_sparsity_args['regularize'].get("alpha_shrinkage_reg", 4),
            "max_pruning_rate": self.channel_sparsity_args['max_pruning_rate'],
            "MAC_params": self.channel_sparsity_args['MAC_params'],
            "isomorphic": self.channel_sparsity_args.get("isomorphic", False)
        }
        self.pruner = None
        self.model_forward_fn = forward_fn
        self.MACs_per_layer: Dict[str, List[float]] = {}
        self.channels_pruner_args["current_round_to"] = 1
        self.config_folder = config_folder
        self.max_imp_current_step = torch.tensor(0.0, device=device)
        self.slice_block_size = None
        self.verbose = self.channel_sparsity_args.get("verbose", 1)

        # Stats collection & compensation state (used by VBP and any criterion with compensation)
        self.vbp_importance = None
        self._compensation_means = None  # collected means for bias compensation (any criterion)
        self.train_loader = train_loader  # passed at init or set externally
        self._vbp_max_batches = self.channel_sparsity_args.get("max_batches", 200)
        self._vbp_var_loss_weight = self.channel_sparsity_args.get("var_loss_weight", 0.0)
        self._vbp_norm_per_layer = self.channel_sparsity_args.get("norm_per_layer", False)
        self._no_compensation = self.channel_sparsity_args.get("no_compensation", False)
        self._stats_fresh = False  # True after stats collected, False after prune step

        # Unified pipeline state
        self._epochs_ft = self.channel_sparsity_args.get("epochs_ft", 0)
        self._model_changed = False
        self._reparam_manager = None
        self._reparam_lambda = self.channel_sparsity_args.get("reparam_lambda", 0.01)
        self._post_stats_hook = None  # callable(model) — called after stats collection (e.g., DDP sync)

        # Build ignored_layers (needed before MAC estimation)
        self.set_layers_to_prune(model)

        # Compute iterative_steps from epoch schedule
        self.iterative_steps = (self.end_epoch - self.start_epoch) // self.epoch_rate + 1

        # If mac_target is set, compute channel ratio analytically
        if self.mac_target > 0:
            self.global_prune_rate = self._estimate_channel_ratio(model)
            _log(log, f"MAC target {self.mac_target:.3f} -> channel prune rate {self.global_prune_rate:.4f}")

        # Pruning schedule (geometric vs linear)
        self.pruning_schedule = channel_sparsity_args.get('pruning_schedule', 'linear')
        self._total_steps = self.iterative_steps
        self._geometric_step = 0
        if self.pruning_schedule == 'geometric':
            total_keep = 1.0 - self.global_prune_rate
            self._per_step_prune = 1.0 - (total_keep ** (1.0 / self._total_steps))
            self.iterative_steps = 1  # single-step pruner, reinit after each
            self._total_prune_rate = self.global_prune_rate
            self.global_prune_rate = self._per_step_prune

        # BN recalibration after pruning steps
        self._bn_recalibration = channel_sparsity_args.get('bn_recalibration', False)

        # Sparse pre-training config
        self._sparse_mode = channel_sparsity_args.get('sparse_mode', 'none')
        self._sparse_gmp_target = channel_sparsity_args.get('sparse_gmp_target', 0.5)
        self._sparse_modules = []
        if self._sparse_mode != 'none':
            for name, m in model.named_modules():
                if name.replace("module.", "") in self.layers_to_prune:
                    self._sparse_modules.append(m)

        # Initial MAC measurement
        self._original_macs = count_ops_and_params(model, self.example_inputs)[0]
        _ = self.measure_macs_masked_model(model)

        # Skip stats at init when sparse pre-training precedes pruning
        # (stats will be stale by the first prune epoch)
        skip_stats = (self._sparse_mode != 'none' and self.start_epoch > 0)
        self.init_channel_pruner(model, log, print_layers=True,
                                 collect_stats=not skip_stats)

    @property
    def total_epochs(self):
        """Total epochs across all phases: sparse + PAT + FT."""
        return self.end_epoch + 1 + self._epochs_ft

    @property
    def model_changed(self):
        """True if model structure changed since last check. Auto-resets."""
        changed = self._model_changed
        self._model_changed = False
        return changed

    @property
    def phase(self):
        """Current training phase based on epoch."""
        if self._sparse_mode != 'none' and self.current_epoch < self.start_epoch:
            return "Sparse"
        elif self.prune_channels:
            return "PAT"
        else:
            return "FT"

    def init_channel_pruner(self, model, log=None, print_layers=False, collect_stats=True,
                             train_loader=None):
        """Build importance criterion, pruner, and optionally collect stats.

        Called once from __init__ (with collect_stats=True) and again after each
        geometric step (with collect_stats=False to reuse cached importance).
        When train_loader is available (via parameter or self.train_loader),
        collects VBP stats + compensation means and sets _stats_fresh=True.
        """
        log = log or self.log
        loader = train_loader or self.train_loader

        # set layers to pruned and their pruning rate
        pruning_ratio_dict = self.set_layers_to_prune(model)

        # set pruning method (importance + pruner entry — no stats yet)
        if self.pruning_method == PruningMethod.BN_SCALE:
            imp = tp.importance.BNScaleImportance()
            pruner_entry = partial(tp.pruner.BNScalePruner, group_lasso=True)
        elif self.pruning_method == PruningMethod.GROUP_NORM:
            imp = tp.importance.GroupMagnitudeImportance(p=2)
            pruner_entry = partial(tp.pruner.GroupNormPruner)
        elif self.pruning_method == PruningMethod.MAC_AWARE:
            L_MACs = {k: v[0] for k, v in self.MACs_per_layer.items()}
            imp = tp.importance.MACAwareImportance(p=2, layers_mac=L_MACs,
                                     params=self.channels_pruner_args["MAC_params"],
                                     current_max=self.max_imp_current_step)
            pruner_entry = partial(tp.pruner.GroupNormPruner)
        elif self.pruning_method == PruningMethod.VBP:
            if not collect_stats and self.vbp_importance is not None:
                imp = self.vbp_importance
            else:
                imp = tp.importance.VarianceImportance(
                    norm_per_layer=self._vbp_norm_per_layer)
                self.vbp_importance = imp
            pruner_entry = partial(tp.pruner.VBPPruner)
        elif self.pruning_method == PruningMethod.LAMP:
            imp = tp.importance.LAMPImportance(p=2)
            pruner_entry = partial(tp.pruner.GroupNormPruner)
        elif self.pruning_method == PruningMethod.RANDOM:
            imp = tp.importance.RandomImportance()
            pruner_entry = partial(tp.pruner.GroupNormPruner)
        else:
            raise NameError(f'Unsupported pruner method. {self.channels_pruner_args["pruning_method"]}')

        # Create pruner first (builds DG for graph-walking)
        grad_d = {}
        for n, m in model.named_parameters():
            grad_d[n] = m.requires_grad
        pruner_kwargs = dict(
            example_inputs=self.example_inputs,
            importance=imp,
            ignored_layers=self.ignored_layers,
            pruning_ratio=self.global_prune_rate,
            pruning_ratio_dict=pruning_ratio_dict if not self.channels_pruner_args["global_pruning"] else None,
            global_pruning=self.channels_pruner_args["global_pruning"],
            round_to=self.channels_pruner_args["current_round_to"],
            max_pruning_ratio=self.channels_pruner_args["max_pruning_rate"],
            forward_fn=self.model_forward_fn,
            isomorphic=self.channels_pruner_args["isomorphic"],
            iterative_steps=self.iterative_steps,
        )
        if self.pruning_method == PruningMethod.VBP:
            pruner_kwargs["verbose"] = self.verbose > 0
        else:
            pruner_kwargs["reg"] = self.channels_pruner_args["reg"]
        self.pruner = pruner_entry(model, **pruner_kwargs)

        # Auto-detect target layers from DG (works for CNN, ViT, ConvNeXt, etc.)
        from torch_pruning.pruner.importance import build_target_layers, collect_activation_means
        target_layers = build_target_layers(model, self.pruner.DG)

        # Collect stats using DG-detected target layers
        if collect_stats and loader is not None:
            if self.pruning_method == PruningMethod.VBP and len(imp.variance) == 0:
                imp.collect_statistics(
                    model, loader, self.device,
                    target_layers=target_layers,
                    max_batches=self._vbp_max_batches)
                if not self._no_compensation:
                    self._compensation_means = dict(imp.means) if imp.means else None
                self._stats_fresh = True
            elif not self._no_compensation and self._compensation_means is None:
                self._compensation_means = collect_activation_means(
                    model, loader, self.device,
                    target_layers=target_layers,
                    max_batches=self._vbp_max_batches)
                _log(log, f"Collected activation means for compensation ({len(self._compensation_means)} layers)")
                self._stats_fresh = True

        # Sync stats across ranks (e.g., DDP broadcast from rank 0)
        if self._post_stats_hook is not None and self._stats_fresh:
            self._post_stats_hook(model)

        # Set mean_dict on pruner for bias compensation
        if not self._no_compensation and self._compensation_means:
            self.pruner.set_mean_dict(self._compensation_means)

        for n, m in model.named_parameters():
            m.requires_grad = grad_d[n]

        # init regularizer
        self.pruner.update_regularizer()

        if self.verbose > 0:
            total_steps = self._total_steps if self.pruning_schedule == 'geometric' else self.iterative_steps
            total_rate = self._total_prune_rate if self.pruning_schedule == 'geometric' else self.global_prune_rate
            _log(log, f"Pruning setup: {total_steps} steps from epoch {self.start_epoch} to {self.end_epoch} ({self.pruning_schedule})")
            _log(log, f"Target prune rate: {total_rate:.3f}, Algorithm: {self.channels_pruner_args['pruning_method']}.")
            if self.pruning_schedule == 'geometric':
                _log(log, f"Geometric per-step prune: {self._per_step_prune:.4f}")
            if self.mac_target > 0:
                _log(log, f"MAC target: {self.mac_target:.3f}")
            if self.channels_pruner_args["round_to"] > 1:
                _log(log, f"Target round_to: {self.channels_pruner_args['round_to']}, Current round_to: {self.channels_pruner_args['current_round_to']}")
            if print_layers:
                num_groups = 0
                source_convs = []
                _prunable = (torch.nn.Conv2d, torch.nn.Linear)
                _log(log, "*************")
                for group in self.pruner.DG.get_all_groups(ignored_layers=self.pruner.ignored_layers,
                                                           root_module_types=self.pruner.root_module_types):
                    _log(log, f"Group {num_groups}:")
                    _log(log, f"Source layer: {group[0].dep.source.name}")
                    source_convs.append(group[0].dep.source.name.split(" ")[0])
                    if any([isinstance(_gt.dep.layer, _prunable) for _gt in group]):
                        _log(log, "Dependencies:")
                        for _g in group:
                            if isinstance(_g.dep.layer, _prunable):
                                _log(log, str(_g.dep)[str(_g.dep).index("=>") + 2:].strip())
                    _log(log, "*************\n")
                    num_groups += 1
                _log(log, f"There are {num_groups} groups of layers, with the following source layers:\n{source_convs}")

            # Viz graph — all 3 views (computational, dependency, both) in png + pdf
            viz_dir = os.path.join(self.config_folder, "viz")
            for fmt in ("png", "pdf"):
                tp.utils.visualize_all_views(
                    self.pruner.DG, viz_dir, format=fmt,
                    ignored_layers=self.ignored_layers)

    def prune(self, model, epoch, log=None, mask_only=True, train_loader=None):
        """Prune the model using TP's built-in iterative_steps.

        Supports two modes: physical pruning (VBP) and mask-with-zeros.
        The pruner is created once at init; each pruning epoch calls step().
        For geometric schedule, reinits pruner after each step (new DG).

        Args:
            train_loader: Optional dataloader for stats re-collection. Falls
                back to self.train_loader if not provided.
        """
        log = log or self.log
        self.current_epoch = epoch

        if not self.prune_channels:
            self.update_channel_mask_dict(model)
            return

        is_vbp = self.pruning_method == PruningMethod.VBP
        loader = train_loader or self.train_loader

        # Reparam sparse pre-training lifecycle
        if self._sparse_mode == "reparam" and epoch < self.start_epoch:
            if self._reparam_manager is None:
                from torch_pruning.utils.reparam import MeanResidualManager
                self._reparam_manager = MeanResidualManager(
                    model, self.layers_to_prune, self.device,
                    lambda_reg=self._reparam_lambda,
                    max_batches=self._vbp_max_batches)
                self._reparam_manager.reparameterize(loader)
                self._model_changed = True
            return

        # Merge reparam back at pruning start
        if self._reparam_manager is not None and self._reparam_manager.is_active:
            self._reparam_manager.merge_back()
            self._model_changed = True
            self.init_channel_pruner(model, log, collect_stats=True)

        # GMP masking for sparse pre-training (before start_epoch)
        if self._sparse_mode == "gmp" and epoch < self.start_epoch:
            from torch_pruning.utils.sparse_utils import gmp_sparsity_schedule, apply_unstructured_pruning
            target_s = gmp_sparsity_schedule(epoch, self.start_epoch, target_s=self._sparse_gmp_target)
            apply_unstructured_pruning(self._sparse_modules, target_s)
            return

        # Bake GMP masks at start_epoch boundary
        if self._sparse_mode == "gmp" and epoch == self.start_epoch:
            from torch_pruning.utils.sparse_utils import remove_pruning_reparametrization
            remove_pruning_reparametrization(self._sparse_modules)

        # Check if this is a pruning epoch
        is_pruning_epoch = (
            self.start_epoch <= epoch
            and (epoch - self.start_epoch) % self.epoch_rate == 0
        )

        if not is_pruning_epoch:
            if self.verbose > 0 and self.channels_pruner_args["reg"] > 0:
                _log(log, f" Epoch {epoch}, regularization phase")
            return

        # Re-collect stats before each step (critical for PAT correctness).
        # Skip if stats are fresh (just collected at init or previous re-init).
        stats_collected = False
        if loader is not None and not self._stats_fresh:
            from torch_pruning.pruner.importance import build_target_layers, collect_activation_means
            target_layers = build_target_layers(model, self.pruner.DG)

            if is_vbp:
                # VBP: re-collect variance + means
                self.vbp_importance.collect_statistics(
                    model, loader, self.device,
                    target_layers=target_layers,
                    max_batches=self._vbp_max_batches)
                if not self._no_compensation:
                    self._compensation_means = dict(self.vbp_importance.means)
            elif not self._no_compensation and self.pruner.mean_dict is not None:
                # Non-VBP: re-collect means only (model changed since last step)
                self._compensation_means = collect_activation_means(
                    model, loader, self.device,
                    target_layers=target_layers,
                    max_batches=self._vbp_max_batches)
            stats_collected = True
        self._stats_fresh = False

        # Sync stats across ranks (e.g., DDP broadcast from rank 0)
        if stats_collected and self._post_stats_hook is not None:
            self._post_stats_hook(model)

        # Update pruner with (possibly synced) compensation means
        if stats_collected and not self._no_compensation and self._compensation_means:
            self.pruner.set_mean_dict(self._compensation_means)

        has_compensation = self.pruner.mean_dict is not None

        self.update_max_imp()

        use_mask = mask_only and not self.prune_channels_at_init

        # On the last pruning step, force physical channel removal so
        # fine-tuning operates on the structurally smaller model.
        if use_mask:
            if self.pruning_schedule == 'geometric':
                is_last_step = (self._geometric_step == self._total_steps - 1)
            else:
                is_last_step = (self.pruner.current_step == self.iterative_steps - 1)
            if is_last_step:
                use_mask = False
                _log(log, " Last pruning step: switching to physical channel removal")

        # Enable VBP meancheck if applicable (needs forward pass before pruning)
        if is_vbp and not use_mask:
            self.pruner.enable_meancheck(model)
            model.eval()
            with torch.no_grad():
                model(self.example_inputs)

        # Unified interactive loop for all criteria (physical + mask)
        for group in self.pruner.step(interactive=True):
            dep, idxs = group[0]
            if len(idxs) > 0:
                if has_compensation:
                    self.pruner._apply_compensation(group, idxs)
                if use_mask:
                    self.mask_group(group)
                else:
                    group.prune()

                if self.verbose > 0:
                    dep_str = str(dep)
                    mode_str = "Mask" if use_mask else "Prune"
                    comp_str = "+comp" if has_compensation else ""
                    # After physical prune, shape[0] is already reduced
                    total_ch = dep.target.module.weight.shape[0] + (len(idxs) if not use_mask else 0)
                    _log(log, f" {mode_str}{comp_str} {len(idxs)}/{total_ch} "
                              f"channels {dep_str[dep_str.find('on'): dep_str.find('(') - 1]}.")

        if is_vbp and not use_mask:
            self.pruner.disable_meancheck()

        # Signal structural change when physical pruning occurred
        if not use_mask:
            self._model_changed = True

        # BN recalibration after pruning step
        if self._bn_recalibration and self.train_loader is not None:
            _recalibrate_bn(model, self.train_loader, self.device, log)

        # Track step completion and handle schedule
        if self.pruning_schedule == 'geometric':
            self._geometric_step += 1
            if self._geometric_step < self._total_steps:
                self.init_channel_pruner(model, log, collect_stats=False)
            else:
                self.prune_channels = False
        else:
            if self.pruner.current_step >= self.iterative_steps:
                self.prune_channels = False

        # After final step, persist is_prune=False so future loads skip pruning
        if not self.prune_channels:
            config_path = os.path.join(self.config_folder, "pruning_config.json")
            with open(config_path, "r") as f:
                sparsity_args = json.load(f)
            sparsity_args['channel_sparsity_args']['is_prune'] = False
            with open(config_path, 'w') as f:
                json.dump(sparsity_args, f, indent=4)
            _log(log, f" Pruning complete — updated {config_path} (is_prune=False)")

        self.update_channel_mask_dict(model)

        # Log MAC progress
        step_num = self._geometric_step if self.pruning_schedule == 'geometric' else self.pruner.current_step
        total_num = self._total_steps if self.pruning_schedule == 'geometric' else self.iterative_steps
        if not self.prune_channels:
            # After physical pruning: absolute MACs vs stored original
            current_macs = count_ops_and_params(model, self.example_inputs)[0]
            mac_ratio = current_macs / self._original_macs
        else:
            # Mask-only: adjusted_macs/touched_macs covers only layers with
            # zeroed channels. Include untouched layers to get ratio vs original.
            adjusted_macs, touched_macs = self.measure_macs_masked_model(model)
            mac_ratio = (self._original_macs - touched_macs + adjusted_macs) / self._original_macs
        _log(log, f" Step {step_num}/{total_num}: "
                  f"MACs = {mac_ratio:.3f} of original")

    def regularize(self, model):
        if not self.prune_channels:
            return 0.0
        if self.current_epoch > self.end_epoch:
            return 0.0
        # VBP does not use traditional regularization; var loss is handled externally
        if self.pruning_method == PruningMethod.VBP:
            return 0.0
        if self.channels_pruner_args["reg"] == 0:
            return 0.0
        self.update_max_imp()
        return self.pruner.regularize(model, alpha=2 ** self.channels_pruner_args["alpha_shrinkage_reg"])

    def set_layers_to_prune(self, model):
        """Build ignored_layers and per-layer pruning ratio dict.

        The pruning ratio is self.global_prune_rate for all prunable layers.
        TP's iterative_steps scheduler handles splitting it across steps.
        """
        self.ignored_layers = []  # Reset to prevent accumulation across calls
        ltp = self.layers_to_prune
        pruning_ratio_dict = {}

        for name, m in model.named_modules():
            name = name.replace("module.", "")
            # QAT modules
            try:
                if isinstance(m, (torch.ao.nn.qat.modules.conv.Conv2d, torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d)):
                    if name in ltp:
                        pruning_ratio_dict[m] = self.global_prune_rate
                    else:
                        self.ignored_layers.append(m)
                    continue
            except Exception:
                pass
            # Ignore ConvTranspose2d (currently unsupported)
            if isinstance(m, torch.nn.ConvTranspose2d):
                self.ignored_layers.append(m)
                continue
            # Ignore PixelShuffle (currently unsupported)
            if isinstance(m, torch.nn.PixelShuffle):
                self.ignored_layers.append(m)
                continue
            # Linear layers: prune if explicitly listed in config layers, otherwise ignore
            if isinstance(m, torch.nn.Linear):
                if name in ltp:
                    pruning_ratio_dict[m] = self.global_prune_rate
                else:
                    self.ignored_layers.append(m)
                continue
            # Pruning only Conv2d (MHA is currently not supported)
            if not isinstance(m, torch.nn.Conv2d):
                continue
            # Ignore layers with single output channels
            if m.out_channels == 1:
                continue
            # Check if the current layer should be pruned or ignored
            if name in ltp:
                pruning_ratio_dict[m] = self.global_prune_rate
            else:
                self.ignored_layers.append(m)

        return pruning_ratio_dict

    @staticmethod
    def mask_group(group):
        for dep, idxs in group:
            target_layer = dep.target.module
            pruning_fn = dep.handler
            if not isinstance(target_layer, (torch.nn.Conv2d, torch.nn.Linear,
                                              torch.nn.ReLU, torch.nn.PReLU,
                                              torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                continue
            mask = torch.ones_like(dep.target.module.weight)
            has_bias = False
            if pruning_fn in [tp.prune_conv_in_channels, tp.prune_linear_in_channels]:
                mask[:, idxs] = 0
            elif pruning_fn in [tp.prune_conv_out_channels, tp.prune_linear_out_channels]:
                mask[idxs] = 0
                if target_layer.bias is not None:
                    has_bias = True
            elif pruning_fn in [tp.prune_depthwise_conv_in_channels, tp.prune_depthwise_conv_out_channels]:
                target_layer.weight.data[idxs] *= 0
                if target_layer.bias is not None:
                    target_layer.bias.data[idxs] *= 0
            elif pruning_fn in [tp.prune_batchnorm_out_channels, tp.prune_layernorm_out_channels]:
                mask[idxs] = 0
                if target_layer.bias is not None:
                    has_bias = True
            else:
                continue
            target_layer.weight.data *= mask
            if has_bias:
                bias_mask = torch.ones_like(dep.target.module.bias)
                bias_mask[idxs] = 0
                target_layer.bias.data *= bias_mask

    def update_channel_mask_dict(self, model):
        for name, param in model.named_modules():
            name = name.replace("module.", "")
            if isinstance(param, nn.Conv2d):
                pruned_channel_indices = torch.where(torch.sum(param.weight, dim=(1, 2, 3)) == 0)[0]
                r, _ = divmod(pruned_channel_indices.shape[0], self.slice_block_size)
                if r == 0:
                    self.channel_mask_dict[name + '.weight'] = pruned_channel_indices[:r]
                else:
                    self.channel_mask_dict[name + '.weight'] = pruned_channel_indices[-r * self.slice_block_size:]

    def measure_macs_masked_model(self, model):
        # Get per-layer MACs (and params) as dictionaries
        macs_dict = count_ops_and_params(model, self.example_inputs, layer_wise=True)[2]

        total_adjusted_macs = 0.0
        total_macs = 0.0
        ltp = self.set_layers_to_prune(model)

        for name, module in model.named_modules():
            name = name.replace("module.", "")
            if module in macs_dict and isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and module in ltp:
                macs_layer = macs_dict[module]
                weight = module.weight

                if isinstance(module, torch.nn.Conv2d):
                    out_channels, in_channels, _, _ = weight.shape
                    zeros_input = (torch.sum(weight, dim=(0, 2, 3)) == 0).sum().item()
                    zeros_output = (torch.sum(weight, dim=(1, 2, 3)) == 0).sum().item()
                elif isinstance(module, torch.nn.Linear):
                    out_channels, in_channels = weight.shape
                    zeros_input = (torch.sum(weight, dim=0) == 0).sum().item()
                    zeros_output = (torch.sum(weight, dim=1) == 0).sum().item()
                else:
                    continue

                active_in_ratio = (in_channels - zeros_input) / in_channels
                active_out_ratio = (out_channels - zeros_output) / out_channels

                ratio = active_in_ratio * active_out_ratio
                if ratio == 1.0:
                    continue
                total_adjusted_macs += macs_layer * ratio
                total_macs += macs_layer

                self.MACs_per_layer[name] = [macs_layer, macs_layer * ratio]

        # update current round to
        target_round_to = self.channels_pruner_args["round_to"]
        if target_round_to > 1:
            current_mac_reduction = 1 - total_adjusted_macs / total_macs
            self.channels_pruner_args["current_round_to"] = min(target_round_to, max(1, int(((current_mac_reduction / (1 - self.mac_target)) * target_round_to + 1))))

        if total_macs == 0:
            total_adjusted_macs, total_macs = 1., 1.
        return total_adjusted_macs, total_macs

    def _estimate_channel_ratio(self, model):
        """Compute global channel pruning ratio to achieve self.mac_target.

        Classifies each layer's MACs by how many dimensions (in/out) are
        pruned in a global-pruning scenario, then solves:

            mac_target = (U + S1*(1-p) + S2*(1-p)^2) / T

        where U = unpruned MACs, S1 = 1-dim pruned, S2 = 2-dim pruned,
        T = total MACs, and p = channel pruning ratio.
        """
        import math

        macs_dict = count_ops_and_params(model, self.example_inputs, layer_wise=True)[2]

        # Build a temp DG to classify layers by pruned dimensions
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=self.example_inputs)

        pruned_dims = {}  # module → set of {'in', 'out'}
        for group in DG.get_all_groups(
                ignored_layers=self.ignored_layers,
                root_module_types=[nn.Conv2d, nn.Linear]):
            for dep, _ in group:
                module = dep.target.module
                if DG.is_out_channel_pruning_fn(dep.handler):
                    pruned_dims.setdefault(module, set()).add('out')
                elif DG.is_in_channel_pruning_fn(dep.handler):
                    pruned_dims.setdefault(module, set()).add('in')

        # Only count MACs for layers that appear in pruning groups (excludes ignored layers)
        mac_total = sum(macs_dict.get(m, 0) or 0 for m in pruned_dims)
        mac_1dim = sum(macs_dict[m] for m, dims in pruned_dims.items()
                       if len(dims) == 1 and macs_dict.get(m) is not None)
        mac_2dim = sum(macs_dict[m] for m, dims in pruned_dims.items()
                       if len(dims) == 2 and macs_dict.get(m) is not None)

        # Solve quadratic: mac_2dim*q^2 + mac_1dim*q + (mac_unpruned - target) = 0
        # where q = 1-p (keep ratio)
        mac_unpruned = mac_total - mac_1dim - mac_2dim
        target_macs = self.mac_target * mac_total
        c = mac_unpruned - target_macs

        if mac_2dim > 0:
            disc = mac_1dim ** 2 - 4 * mac_2dim * c
            if disc < 0:
                _log(self.log, "WARNING: MAC target unreachable, using max pruning rate")
                return self.channels_pruner_args['max_pruning_rate']
            q = (-mac_1dim + math.sqrt(disc)) / (2 * mac_2dim)
        elif mac_1dim > 0:
            q = -c / mac_1dim
        else:
            return 0.0

        p = 1.0 - q
        p = max(0.0, min(p, self.channels_pruner_args['max_pruning_rate']))
        return p

    def update_max_imp(self):
        if self.pruning_method != PruningMethod.MAC_AWARE:
            return
        self.pruner.importance.current_max = torch.ones_like(self.pruner.importance.current_max)
        for group in self.pruner.DG.get_all_groups(ignored_layers=self.pruner.ignored_layers,
                                                   root_module_types=self.pruner.root_module_types):
            if self.pruner._check_pruning_ratio(group):
                imp = self.pruner.importance(group, act_only=True)
                self.pruner.importance.current_max = torch.max(self.pruner.importance.current_max, imp.max())


class SlicePruning:
    def __init__(self, slice_sparsity_args, model, log=None):
        self.log = log
        if slice_sparsity_args is None:
            _log(self.log, "=> Unable to find a valid slice pruning configuration.")
            self.prune_slices = False
            self.prune_slices_at_init = False
            self.block_size = 8
            return

        self.current_epoch = 0
        self.current_pr = 0
        self.slice_sparsity_args = slice_sparsity_args
        if 'is_prune' in slice_sparsity_args.keys():
            self.prune_slices = self.slice_sparsity_args['is_prune']
        else:
            self.prune_slices = True
        self.start_epoch = self.slice_sparsity_args['start_epoch']
        self.end_epoch = self.slice_sparsity_args['end_epoch']
        self.epoch_rate = self.slice_sparsity_args['epoch_rate']
        self.block_size = slice_sparsity_args['block_size']
        self.prune_rate = slice_sparsity_args['prune_rate']
        self.reg = slice_sparsity_args['reg']
        self.slice_sparsity_args['layers'] = {name.replace("module.", "") + '.weight': self.prune_rate for name, m in model.named_modules() if isinstance(m, nn.Conv2d)}
        self.channel_mask_dict = {}

    def extract_slices(self, name: str, w: torch.nn.parameter.Parameter):
        c_out, c_in, y, x = w.shape  # c_out = number of filters, c_in = number of channels
        B = c_out // self.block_size  # number of blocks
        S = B * c_in * y * x  # number of slices
        # cacluate input and output zeros channels
        input_cm = torch.where(w.sum(dim=(0, 2, 3)) == 0)[0].to('cpu')
        B_indices = torch.arange(S).view(B, c_in, y, x)
        icm_indices = B_indices[:, input_cm, :, :].contiguous().view(-1)
        channel_mask = self.channel_mask_dict[name] if self.channel_mask_dict[name].shape[0] > 0 else None
        if channel_mask is not None:
            channel_mask = channel_mask.to('cpu')
            # unpruned filters
            eff_filters = torch.tensor([i for i in range(c_out) if i not in channel_mask])
            # effective number of blocks
            eff_B = eff_filters.shape[0] // self.block_size
            # indices of unpruned filters
            eff_filter_indices = torch.cat([torch.tensor([b + a for b in range(0, self.block_size * eff_B, eff_B)]) for a in range(eff_B)])
            filter_indices = torch.cat((eff_filters[eff_filter_indices], channel_mask))
            ocm_indices = torch.tensor([i for i in range(int(S * (1 - channel_mask.shape[0] / c_out)), S)])
        else:
            filter_indices = torch.cat([torch.tensor([b + a for b in range(0, self.block_size * B, B)]) for a in range(B)])
            ocm_indices = torch.tensor([], dtype=torch.int64)
        # reordering filters by SNP itterations
        w = w[filter_indices]
        # keep indices of original ordering
        revert_fi = torch.argsort(filter_indices)
        # split the layer into blocks
        Blocks = w.view(B, self.block_size, c_in, y, x)  # [B, A, c_in, y, x]
        # split each block into slices
        Slices = Blocks.permute(0, 2, 3, 4, 1).contiguous()  # [B, c_in, y, x, A]
        Slices = Slices.view(S, self.block_size)  # [S, A]
        # calculate indices of first index of each column
        fc_indices = torch.tensor([s for s in range(S) if s % y == 0])
        # calculate indices of zero input channels
        return Slices, revert_fi, fc_indices, ocm_indices, icm_indices

    def calc_prune_rate(self):
        if self.slice_sparsity_args['pruning_gradually'] and self.current_epoch < self.end_epoch:  # would be move outside of loop
            num_steps = sum([1 for i in range(self.start_epoch, self.end_epoch) if
                             i % self.epoch_rate == 0])
            curr_step = sum([1 for i in range(self.start_epoch, self.current_epoch + 1) if
                             i % self.epoch_rate == 0])
            self.current_pr = self.prune_rate * curr_step / num_steps
        else:
            self.current_pr = self.prune_rate

    def regularize(self, model):
        if not self.prune_slices:
            return torch.tensor(0.0)
        assert self.slice_sparsity_args["pruning_mode"] == "Prune", "sparsity loss is available only on pruned stage"
        SP_loss = 0
        # run over the relevant layers, as defined in the config
        for name, param in model.named_parameters():
            name = name.replace("module.", "")

            if name in self.slice_sparsity_args['layers'].keys():
                Slices, _, fc_indices, _, _ = self.extract_slices(name, param)
                # disable pruning on first index of each column (for loss it's equal to set them as zeros)
                if self.slice_sparsity_args['disable_first_index']:
                    Slices[fc_indices] = 0
                if "L2_norm" in self.slice_sparsity_args['pruning_method']:
                    # calculate the L2 norm of each slice
                    L2_norm_slices = torch.norm(Slices, dim=1)
                    # calculate alpha
                    alpha = (1 / (L2_norm_slices.detach() + 1e-8)).view(-1, 1)
                    Slices = Slices * alpha
                    # calculate L2-norm as a loss
                    L_loss = torch.norm(Slices, dim=1)
                    SP_loss += torch.sum(L_loss)
        if not self.prune_slices or self.slice_sparsity_args["reg"] == 0 or self.current_epoch > self.end_epoch:
            SP_loss = torch.zeros_like(SP_loss)
        return SP_loss * self.reg  # * self.current_pr / self.prune_rate

    def prune(self, model, epoch, log=None):
        log = log or self.log
        self.current_epoch = epoch
        if not self.prune_slices:
            return
        if self.slice_sparsity_args["pruning_mode"] == "Unprune":
            _log(log, "Slice pruning disabled in Unprune mode")
            return

        pruning = (self.start_epoch <= self.current_epoch and self.current_epoch % self.epoch_rate == 0) or self.current_epoch >= self.end_epoch

        if not pruning:
            _log(log, "Slice pruning disabled in current epoch")
            return
        else:
            self.calc_prune_rate()
            _log(log, f"Epoch {self.current_epoch}, slice pruning progress:")
            _log(log, f"Pruning from epoch {self.start_epoch} to epoch {self.end_epoch}, with a current pruning rate of {self.current_pr}.")
            _log(log, f"Total target: {self.prune_rate}.")
            # loop over layers and pruned them
            for name, param in model.named_parameters():
                name = name.replace("module.", "")

                # slice pruning
                if name in self.slice_sparsity_args['layers'].keys():
                    # extract slices and sort them by their L2-norm
                    Slices, revert_indices, fc_indices, ocm_indices, icm_indices = self.extract_slices(name, param)
                    L2_norm_slices = torch.norm(Slices, dim=1)
                    # disable pruning on first index of each column (for loss it's equal to set them as zeros)
                    if self.slice_sparsity_args['disable_first_index']:
                        L2_norm_slices[fc_indices] = torch.inf
                    # disable pruning on slices which already pruned during channel pruning
                    L2_norm_slices[ocm_indices] = torch.inf
                    L2_norm_slices[icm_indices] = torch.inf

                    # Sort slices by their L2-norm
                    sorted_slices_norms, _ = L2_norm_slices.flatten().sort()
                    # Determine the threshold based on p%
                    slices_threshold_index = int(self.current_pr * len(sorted_slices_norms))
                    slices_threshold = sorted_slices_norms[slices_threshold_index]
                    # Create a mask to zero out entries below the threshold
                    slices_mask = L2_norm_slices <= slices_threshold
                    Slices[slices_mask] = 0
                    # prune the layer
                    c_out, c_in, _, _ = param.shape
                    B = c_out // self.block_size  # number of blocks
                    pruned_layer = Slices.view(B, c_in, 3, 3, self.block_size).permute(0, 4, 1, 2, 3).contiguous()
                    # revert ordering of filters
                    pruned_layer = pruned_layer.view(c_out, c_in, 3, 3)[revert_indices]
                    param.data = pruned_layer
                    _log(log, f" Mask {slices_mask.sum()} / {Slices.shape[0]} slices on {name}")

        _log(log, f" Current slice sparsity: {self.calc_current_sparsity(model)}")

    def calc_current_sparsity(self, model):
        with torch.no_grad():
            num_slices = 0
            num_zero_slices = 0
            for name, param in model.named_parameters():
                name = name.replace("module.", "")
                if name in self.slice_sparsity_args['layers'].keys():
                    # icm = torch.where(param.sum(dim=(0, 2, 3)) == 0)
                    # ocm = torch.where(param.sum(dim=(1, 2, 3)) == 0)
                    # param[icm] = torch.ones_like(param[icm])
                    # param[ocm] = torch.zeros_like(param[ocm])
                    Slices, _, _, _, _ = self.extract_slices(name, param)
                    num_zero_slices += torch.sum(torch.sum(Slices, dim=1) == 0)
                    num_slices += Slices.shape[0]
            return num_zero_slices / num_slices


def build_inputs(cfg, device):
    """
    Build dummy input tensors from a config specification.

    Config options:
    - "input_shape": either a single shape [B, C, H, W] or a list of shapes [[...], [...], ...]
    - "container": optional, if set to "tuple" returns a tuple of tensors, otherwise returns a list.

    Returns:
    - torch.Tensor if single shape
    - list[torch.Tensor] if multiple shapes
    - tuple[torch.Tensor, ...] if multiple shapes and container="tuple"
    """
    if "input_shape" not in cfg:
        # Require config to define "input_shape"
        raise ValueError("Config must contain 'inputs'.")

    raw = cfg["input_shape"]
    container = cfg.get("container", None)

    # Case 1: single tensor, shape given as [B, C, H, W]
    if isinstance(raw, (list, tuple)) and all(isinstance(x, int) for x in raw):
        shape = tuple(int(x) for x in raw)
        return torch.zeros(shape, device=device)

    # Case 2: multiple tensors, shape list like [[...], [...], ...]
    if isinstance(raw, (list, tuple)) and all(isinstance(s, (list, tuple)) for s in raw):
        shapes = [tuple(int(x) for x in s) for s in raw]
        tensors = [torch.zeros(s, device=device) for s in shapes]
        if container == "tuple":
            # Return tensors as tuple if requested
            return tuple(tensors)
        return tensors  # default: return as list

    # Invalid input format
    raise ValueError(
        "'inputs' must be [B,C,H,W] or a list of such shapes; "
        "use 'container': 'tuple' to return a tuple.")


# Backward-compatible aliases (PEP 8: classes should use CapWords)
channel_pruning = ChannelPruning
slice_pruning = SlicePruning
