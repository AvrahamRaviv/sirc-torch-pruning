"""VBP (Variance-Based Pruning) pruner with bias compensation.

Reference: https://arxiv.org/pdf/2507.12988
"""

import torch
import torch.nn as nn

from .base_pruner import BasePruner
from .. import function


class VBPPruner(BasePruner):
    def __init__(self, model, example_inputs, *args, mean_dict=None, verbose=True, **kwargs):
        super().__init__(model, example_inputs, *args, **kwargs)
        self.example_inputs = example_inputs
        self.mean_dict = mean_dict
        self.verbose = verbose

        # mean-check state
        self._meancheck_enabled = False
        self._meancheck_handles = []

    @torch.no_grad()
    def set_mean_dict(self, mean_dict):
        self.mean_dict = mean_dict

    @torch.no_grad()
    def enable_meancheck(self, model):
        if self._meancheck_enabled:
            return

        def _cache_input(mod, inp, out):
            if inp and inp[0] is not None:
                mod._last_input = inp[0].detach()

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self._meancheck_handles.append(m.register_forward_hook(_cache_input))

        self._meancheck_enabled = True

    @torch.no_grad()
    def disable_meancheck(self):
        for h in self._meancheck_handles:
            h.remove()
        self._meancheck_handles.clear()
        self._meancheck_enabled = False

    @torch.no_grad()
    def _per_channel_mean(self, x):
        # x is input to Conv/Linear:
        # Conv input:  [N, C, H, W] -> mean over N,H,W -> [C]
        # ViT Linear:  [N, T, C]    -> mean over N,T   -> [C]
        # Linear:      [N, C]       -> mean over N     -> [C]
        if x is None:
            return None
        if x.dim() == 4:
            return x.mean(dim=(0, 2, 3))
        if x.dim() == 3:
            return x.mean(dim=(0, 1))
        if x.dim() == 2:
            return x.mean(dim=0)
        return None

    @torch.no_grad()
    def step(self, interactive=False, enable_compensation=True):
        self.current_step += 1

        if interactive:
            return self._prune()

        # mean_dict gates compensation: if provided (non-None), apply bias compensation.
        if self.mean_dict is None:
            for group in self._prune():
                group.prune()
            return

        for _gi, group in enumerate(self._prune()):
            dep0, idxs = group[0]
            if len(idxs) == 0:
                continue
            root = dep0.target.module

            # Optional mean-check diagnostic (requires enable_meancheck + forward pass)
            consumer = None
            mu_before = None
            mu_after = None
            if self._meancheck_enabled:
                for dep, _ in group[1:]:
                    if dep.handler in (
                            function.prune_conv_in_channels,
                            function.prune_linear_in_channels,
                    ):
                        if isinstance(dep.target.module, (nn.Conv2d, nn.Linear)):
                            consumer = dep.target.module
                            break
                if consumer is not None:
                    mu_before = self._collect_out_mean(self.model, consumer, self.example_inputs)

            if enable_compensation:
                self._apply_compensation(group, idxs)

            dep, idxs = group[0]
            dep_str = str(dep)
            if self.verbose:
                idxs_ratio_str = f"{len(idxs)} / {dep.target.module.weight.shape[0]}"
                log_str = f"Prune {idxs_ratio_str} channels {dep_str[dep_str.find('on'): dep_str.find('(') - 1]}."
                print(f"[VBP-prune] {log_str}")
            group.prune()

            if self._meancheck_enabled and consumer is not None:
                mu_after = self._collect_out_mean(self.model, consumer, self.example_inputs)
                if mu_before is not None and mu_after is not None and self.verbose:
                    C = min(len(mu_before), len(mu_after))
                    d = (mu_after[:C] - mu_before[:C]).abs()
                    print(
                        f"[VBP-meancheck] {consumer}: "
                        f"mean(|Δμ|)={d.mean():.5f}, "
                        f"max(|Δμ|)={d.max():.5f}"
                    )

    @torch.no_grad()
    def _add_bias(self, module, delta):
        if module.bias is None:
            module.bias = nn.Parameter(delta.clone())
        else:
            module.bias.data += delta

    @torch.no_grad()
    def _apply_compensation(self, group, idx_to_prune):
        """Compensate consumer biases for pruned channels using calibration means.

        NOTE on residual stream pruning (e.g. ResNet conv3 -> add -> next block):
        When pruning channels that feed through a residual add, the compensation
        uses the post-BN+ReLU mean of the pruned conv's output. This is an
        approximation — the true input to the consumer is (residual + conv_output),
        but we only have the conv_output mean. Fine-tuning recovers this gap.
        Interior block pruning (conv1/conv2 in Bottleneck) is exact since there
        is no add node between the pruned conv and its consumer.
        """
        dep0, _ = group[0]
        root = dep0.target.module  # pruned Conv / Linear

        # Use calibration means from mean_dict (keyed by root module)
        mu = self.mean_dict.get(root)
        if mu is None:
            if self.verbose:
                print(f"[VBP] No calibration mean for {root}, skipping compensation")
            return

        rem = torch.as_tensor(idx_to_prune, device=root.weight.device)
        mu = mu.to(root.weight.device)
        compensated = False

        for dep, _ in group[1:]:
            handler = dep.handler
            consumer = dep.target.module

            # We only compensate real parameterized consumers
            if not isinstance(consumer, (nn.Conv2d, nn.Linear)):
                continue

            # ----- Linear consumer -----
            if handler == function.prune_linear_in_channels:
                # consumer.weight: [C_out, C_in]
                W = consumer.weight[:, rem]  # [C_out, |rem|]
                delta_b = (W * mu[rem]).sum(dim=1)  # [C_out]
                self._add_bias(consumer, delta_b)
                compensated = True
                if self.verbose:
                    print(f"[VBP-comp] consumer {consumer} compensated (calibration mu)")

            # ----- Conv consumer -----
            elif handler == function.prune_conv_in_channels:
                if consumer.groups == consumer.in_channels:
                    # depthwise
                    delta_b = torch.zeros(consumer.out_channels, device=consumer.weight.device)
                    delta_b[rem] = consumer.weight[rem, 0].sum(dim=(1, 2)) * mu[rem]
                    self._add_bias(consumer, delta_b)
                else:
                    W = consumer.weight[:, rem, :, :]  # [C_out, |rem|, kH, kW]
                    W_sp = W.sum(dim=(2, 3))  # [C_out, |rem|]
                    delta_b = (W_sp * mu[rem]).sum(dim=1)  # [C_out]
                    self._add_bias(consumer, delta_b)

                compensated = True
                if self.verbose:
                    print(f"[VBP-comp] consumer {consumer} compensated (calibration mu)")

        if not compensated and self.verbose:
            print(f"[VBP] No compensation applied for {root} (no param consumer found)")

    @torch.no_grad()
    def _collect_out_mean(self, model, module, x):
        handles = []
        out_buf = {}

        def hook(m, inp, out):
            if out.dim() == 4:
                out_buf["mu"] = out.mean(dim=(0, 2, 3)).detach().cpu()
            elif out.dim() == 3:
                out_buf["mu"] = out.mean(dim=(0, 1)).detach().cpu()
            elif out.dim() == 2:
                out_buf["mu"] = out.mean(dim=0).detach().cpu()

        handles.append(module.register_forward_hook(hook))
        model(x)
        for h in handles:
            h.remove()
        return out_buf.get("mu", None)
