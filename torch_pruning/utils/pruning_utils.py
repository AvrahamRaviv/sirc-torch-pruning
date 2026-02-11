import os
import json
import time
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


class Pruning:
    """
    High-level pruning interface combining channel and slice pruning.
    """
    def __init__(self, model: nn.Module, config_folder: str, forward_fn: Optional[Any] = None,
                 log: Optional[Any] = None, device: Optional[torch.device] = None) -> None:
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

        self.channel_pruner = ChannelPruning(channel_sa, model, config_folder, forward_fn, log, device)
        self.slice_pruner = SlicePruning(slice_sa, model, log)
        # Synchronize slice block size and channel mask dictionary between pruners.
        self.channel_pruner.slice_block_size = self.slice_pruner.block_size
        self.slice_pruner.channel_mask_dict = self.channel_pruner.channel_mask_dict

    def channel_regularize(self, model: nn.Module) -> None:
        """Apply channel regularization to the model."""
        self.channel_pruner.regularize(model)

    def slice_regularize(self, model: nn.Module) -> torch.Tensor:
        """Apply slice regularization to the model."""
        return self.slice_pruner.regularize(model)

    def prune(self, model: nn.Module, epoch: int, log: Optional[Any] = None,
              mask_only: bool = True) -> None:
        """Perform pruning using both channel and slice pruners."""
        self.channel_pruner.prune(model, epoch, log=log, mask_only=mask_only)
        self.slice_pruner.prune(model, epoch, log=log)


class ChannelPruning:
    """
    Implements channel pruning using a chosen pruning method.
    """
    def __init__(self, channel_sparsity_args: Optional[Dict[str, Any]], model: nn.Module,
                 config_folder: str, forward_fn: Any, log: Optional[Any] = None,
                 device: Optional[torch.device] = None) -> None:
        self.channel_mask_dict: Dict[str, torch.Tensor] = {}
        if channel_sparsity_args is None:
            _log(log, "=> Unable to find a valid channel pruning configuration.")
            self.prune_channels = False
            self.prune_channels_at_init = False
            self.reach_mac_target = False
            return

        self.prune_channels = channel_sparsity_args.get('is_prune', True)
        self.channel_sparsity_args = channel_sparsity_args
        self.device = device
        self.infer = self.channel_sparsity_args['infer']

        # Example inputs for forward pass estimation.
        self.example_inputs = build_inputs(channel_sparsity_args, device=device)
        if self.infer:
            self.example_inputs = (self.example_inputs[0][:, :1], self.example_inputs[0][:, :1])
        self.current_epoch: int = 0
        self.current_step = None
        self.current_pr: float = 0.0
        self.ignored_layers: List[Any] = []
        self.start_epoch: int = self.channel_sparsity_args['start_epoch']
        self.end_epoch: int = self.channel_sparsity_args['end_epoch']
        self.epoch_rate: int = self.channel_sparsity_args['epoch_rate']
        self.global_prune_rate: float = self.channel_sparsity_args['global_prune_rate']
        self.layers_to_prune = self.channel_sparsity_args['layers']
        self.mac_target: float = self.channel_sparsity_args.get('mac_target', 0.0)
        self.reach_mac_target = self.channel_sparsity_args.get("reach_mac_target", False)
        self.log_str = self.channel_sparsity_args.get("log_str", None)
        self.prune_channels_at_init = self.channel_sparsity_args['prune_channels_at_init'] or self.infer

        self.channels_pruner_args = {
            "pruning_method": self.channel_sparsity_args['pruning_method'],
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
        self.regularization = None
        self.model_forward_fn = forward_fn
        self.init_channel_mask: Dict[Any, Any] = {}
        self.MACs_per_layer: Dict[str, List[float]] = {}
        self.channels_pruner_args["current_round_to"] = 1
        _ = self.measure_macs_masked_model(model)
        self.config_folder = config_folder
        self.max_imp_current_step = torch.tensor(0.0, device=device)
        self.slice_block_size = None
        self.prune_at_target = self.channel_sparsity_args.get("prune_at_target", False)
        self.reset_optimizer = False
        self.verbose = self.channel_sparsity_args.get("verbose", 1)

        # VBP-specific state
        self.vbp_importance = None
        self.train_loader = None  # set externally for VBP stats collection
        self._vbp_model_type = self.channel_sparsity_args.get("model_type", None)
        self._vbp_max_batches = self.channel_sparsity_args.get("max_batches", 200)
        self._vbp_var_loss_weight = self.channel_sparsity_args.get("var_loss_weight", 0.0)
        self._vbp_norm_per_layer = self.channel_sparsity_args.get("norm_per_layer", False)

        self.init_channel_pruner(model, log, print_layers=True)

    def init_channel_pruner(self, model, log=None, print_layers=False):
        # set layers to pruned and their pruning rate
        pruning_ratio_dict = self.set_layers_to_prune(model)

        # set pruning method
        if self.channels_pruner_args['pruning_method'] == 'BNScalePruner':
            imp = tp.importance.BNScaleImportance()
            pruner_entry = partial(tp.pruner.BNScalePruner, group_lasso=True)
        elif self.channels_pruner_args['pruning_method'] == 'GroupNormPruner':
            imp = tp.importance.GroupNormImportance(p=2, gamma=self.channels_pruner_args["gamma_reg"])
            pruner_entry = partial(tp.pruner.GroupNormPruner)
        elif self.channels_pruner_args['pruning_method'] == 'MACAwareImportance':
            L_MACs = {k: v[0] for k, v in self.MACs_per_layer.items()}
            imp = tp.importance.MACAwareImportance(p=2, layers_mac=L_MACs,
                                     params=self.channels_pruner_args["MAC_params"],
                                     current_max=self.max_imp_current_step)
            pruner_entry = partial(tp.pruner.GroupNormPruner)
        elif self.channels_pruner_args['pruning_method'] == 'VBP':
            imp = tp.importance.VarianceImportance(
                norm_per_layer=self._vbp_norm_per_layer)
            # Collect stats if train_loader is available and stats not yet collected
            if self.train_loader is not None and len(imp.variance) == 0:
                target_layers = self._build_target_layers(model)
                imp.collect_statistics(
                    model, self.train_loader, self.device,
                    target_layers=target_layers,
                    max_batches=self._vbp_max_batches)
            self.vbp_importance = imp
            pruner_entry = partial(tp.pruner.VBPPruner, mean_dict=imp.means)
        else:
            raise NameError(f'Unsupported pruner method. {self.channels_pruner_args["pruning_method"]}')

        # init pruner
        grad_d = {}
        for n, m in model.named_parameters():
            grad_d[n] = m.requires_grad
        pruner_kwargs = dict(
            example_inputs=self.example_inputs,
            importance=imp,
            ignored_layers=self.ignored_layers,
            pruning_ratio=self.current_pr,
            pruning_ratio_dict=pruning_ratio_dict if not self.channels_pruner_args["global_pruning"] else None,
            global_pruning=self.channels_pruner_args["global_pruning"],
            round_to=self.channels_pruner_args["current_round_to"],
            max_pruning_ratio=self.channels_pruner_args["max_pruning_rate"],
            forward_fn=self.model_forward_fn,
            isomorphic=self.channels_pruner_args["isomorphic"],
        )
        # VBPPruner/BasePruner doesn't accept reg; only pass for regularization-based pruners
        if self.channels_pruner_args['pruning_method'] == 'VBP':
            pruner_kwargs["verbose"] = self.verbose > 0
        else:
            pruner_kwargs["reg"] = self.channels_pruner_args["reg"]
        self.pruner = pruner_entry(model, **pruner_kwargs)
        for n, m in model.named_parameters():
            m.requires_grad = grad_d[n]

        # init regularizer
        self.pruner.update_regularizer()

        if self.verbose > 0:
            _log(log, f"Epoch {self.current_epoch}, pruning progress:")
            _log(log, f"Pruning from epoch {self.start_epoch} to epoch {self.end_epoch}, with a current pruning rate of {self.current_pr:.3f}.")
            _log(log, f"Total target: {self.global_prune_rate}, Pruning algorithm: {self.channels_pruner_args['pruning_method']}.")
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

            # Viz graph
            tp.utils.visualize_graph(self.pruner.DG, self.config_folder, show_groups=True)

    def prune(self, model, epoch, log=None, mask_only=True, step=None):
        """
            Prune the model
            We are supporting two modes: 1. Physically prune channels. 2. Mask with zeros.
            It is handaled by mask_only flag.
        """
        if not self.prune_channels:
            self.update_channel_mask_dict(model)
            if self.reach_mac_target and self.verbose > 0:
                _log(log, f" Model already reach {self.mac_target} MACs reduction")
                _log(log, " Pruning statistics:")
                for line in self.log_str.split("\n")[:-1]:
                    _log(log, f" {line}")
            return
        self.current_epoch = epoch
        self.log_str = ""
        is_vbp = self.channels_pruner_args['pruning_method'] == 'VBP'
        if self.start_epoch <= epoch and (not is_vbp or epoch % self.epoch_rate == 0):
            self.current_step = step
            self.init_channel_pruner(model, log)
            self.update_max_imp()

            if is_vbp:
                # VBP: use VBPPruner.step() which handles compensation internally
                self.pruner.enable_meancheck(model)
                model.eval()
                with torch.no_grad():
                    model(self.example_inputs)
                self.pruner.step(interactive=False, enable_compensation=True)
                self.pruner.disable_meancheck()
                self.log_str += "VBP pruning with compensation applied\n"
                _log(log, " VBP pruning with compensation applied")
            else:
                for group in self.pruner.step(interactive=True):
                    dep, idxs = group[0]
                    dep_str = str(dep)
                    if len(idxs) > 0:
                        mask_only = mask_only and not self.prune_channels_at_init and not self.reach_mac_target
                        pom = ["Mask", "masked"] if mask_only else ["Prune", "pruned"]
                        idxs_ratio_str = f"{len(idxs)} / {dep.target.module.weight.shape[0]}"
                        log_str = f"{pom[0]} {idxs_ratio_str} channels {dep_str[dep_str.find('on'): dep_str.find('(') - 1]}."
                        if self.verbose > 1:
                            log_str += f" Indices of {pom[1]} channels are: {idxs}."
                        _log(log, f" {log_str}")
                        self.log_str += f"{log_str}\n"

                        if mask_only:
                            self.mask_group(group)
                        else:
                            group.prune(idxs[:len(idxs) - (len(idxs) % self.slice_block_size)])
        elif self.verbose > 0 and self.channels_pruner_args["reg"] > 0:
            _log(log, f" Epoch {self.current_epoch}, regularization phase with alpha = {self.channels_pruner_args['reg']}")

        if is_vbp:
            # VBP physically prunes but supports iterative PAT — only stop after
            # all scheduled epochs are done (current_epoch >= end_epoch)
            if self.current_epoch >= self.end_epoch:
                self.prune_channels = False
        elif self.prune_channels_at_init or not mask_only or self.reach_mac_target:
            self.prune_channels = False

        self.update_channel_mask_dict(model)

        # log MACs
        current_macs, total_macs = self.measure_macs_masked_model(model)
        if mask_only or self.prune_channels_at_init:
            _log(log, f" Current MACs are {current_macs / total_macs:.3f}% of original model")
            if current_macs / total_macs < self.mac_target and self.prune_at_target:
                self.reach_mac_target = True
                # update the config file: 1. reach_mac_target, 2. logs of pruning
                config_path = os.path.join(self.config_folder, "pruning_config.json")
                with open(config_path, "r") as f:
                    sparsity_args = json.load(f)
                    time.sleep(5)
                sparsity_args['channel_sparsity_args']['reach_mac_target'] = True
                if self.prune_at_target:
                    self.log_str = self.log_str.replace("Mask", "Prune")
                    self.reset_optimizer = True
                sparsity_args['channel_sparsity_args']['log_str'] = f"reach_mac_target at epoch {self.current_epoch}\n" + self.log_str
                with open(config_path, 'w') as file:
                    json.dump(sparsity_args, file, indent=4)

        else:
            _log(log, f" Model already reach {self.mac_target} MACs reduction")

    def regularize(self, model):
        # VBP does not use traditional regularization; var loss is handled externally
        if self.channels_pruner_args['pruning_method'] == 'VBP':
            return
        if not self.prune_channels or self.channels_pruner_args["reg"] == 0 or self.current_epoch > self.end_epoch:
            return
        self.update_max_imp()
        self.pruner.regularize(model, alpha=2 ** self.channels_pruner_args["alpha_shrinkage_reg"])

    def calc_prune_rate(self):
        """
        Calculate the pruning rate by interpolating between the coarse epoch-level values
        using the current step progress within the epoch.
        """
        if self.current_epoch < self.start_epoch:
            self.current_pr = 0
        elif self.current_epoch < self.end_epoch:
            # Calculate total coarse updates (epochs where pruning is updated)
            num_coarse_steps = sum([1 for i in range(self.start_epoch, self.end_epoch)
                                   if i % self.epoch_rate == 0])
            # Count how many coarse updates have occurred up to the beginning of the current epoch
            current_coarse_index = sum([1 for i in range(self.start_epoch, self.current_epoch)
                                       if i % self.epoch_rate == 0])
            # Coarse pruning values at the current and next update points
            prev_pr = self.global_prune_rate * current_coarse_index / num_coarse_steps
            next_pr = self.global_prune_rate * (current_coarse_index + 1) / num_coarse_steps

            if self.current_step is None:
                self.current_pr = next_pr
            else:
                # Interpolate using the current step progress within the epoch
                step_fraction = self.current_step[0] / self.current_step[1] if self.current_step[1] > 0 else 0
                self.current_pr = prev_pr + step_fraction * (next_pr - prev_pr)
        else:
            self.current_pr = self.global_prune_rate

    def set_layers_to_prune(self, model):
        """
            Function that setting layers to prune and their coresponding pruning rate
            As of now it is kind of hard coded, will be replaced by self.channel_sparsity_args['cp_layers'] style (as in Slice Pruning)
        """
        self.calc_prune_rate()
        ltp = self.layers_to_prune
        pruning_ratio_dict = {}

        for name, m in model.named_modules():
            name = name.replace("module.", "")
            # Check spyk first
            try:
                if isinstance(m, (torch.ao.nn.qat.modules.conv.Conv2d, torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d)):
                    if name in ltp:
                        pruning_ratio_dict[m] = self.current_pr
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
            # Linear layers: VBP prunes specific Linears, others ignore all
            if isinstance(m, torch.nn.Linear):
                if self.channels_pruner_args['pruning_method'] == 'VBP' and name in ltp:
                    pruning_ratio_dict[m] = self.current_pr
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
                pruning_ratio_dict[m] = self.current_pr
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

    def _build_target_layers(self, model):
        """Auto-detect target layers for VBP stats collection based on model_type."""
        model_type = self._vbp_model_type
        if model_type == "vit":
            # HuggingFace ViT: block.intermediate.dense + block.intermediate.intermediate_act_fn
            target_layers = []
            for name, m in model.named_modules():
                if name.endswith(".intermediate.dense"):
                    # Navigate to parent block to get act fn
                    parts = name.split(".")
                    # intermediate.dense -> go up to block level
                    block = model
                    for p in parts[:-1]:  # up to "intermediate"
                        block = getattr(block, p)
                    act_fn = getattr(block, "intermediate_act_fn", None)
                    target_layers.append((m, act_fn))
            return target_layers if target_layers else None
        elif model_type == "convnext":
            target_layers = []
            for name, m in model.named_modules():
                if hasattr(m, "pwconv1") and hasattr(m, "act"):
                    def _make_post_gelu_nchw(act_fn):
                        def fn(x):
                            return act_fn(x).permute(0, 3, 1, 2)
                        return fn
                    target_layers.append((m.pwconv1, _make_post_gelu_nchw(m.act)))
            return target_layers if target_layers else None
        return None

    def update_max_imp(self):
        if self.channels_pruner_args['pruning_method'] != 'MACAwareImportance':
            return
        self.pruner.importance.current_max = torch.ones_like(self.pruner.importance.current_max)
        for group in self.pruner.DG.get_all_groups(ignored_layers=self.pruner.ignored_layers,
                                                   root_module_types=self.pruner.root_module_types):
            if self.pruner._check_pruning_ratio(group):
                imp = self.pruner.importance(group, act_only=True)
                self.pruner.importance.current_max = torch.max(self.pruner.importance.current_max, imp.max())


class SlicePruning:
    def __init__(self, slice_sparsity_args, model, log=None):
        if slice_sparsity_args is None:
            _log(log, "=> Unable to find a valid slice pruning configuration.")
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
            return torch.tensor(0).to('cuda')
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
