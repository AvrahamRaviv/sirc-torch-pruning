import abc
import torch
import torch.nn as nn

import typing
from . import function
from ..dependency import Group
from ..dependency.index_mapping import _FlattenIndexMapping
from .. import ops
import math
import numpy as np
from collections import OrderedDict
from ..utils.compute_mat_grad import ComputeMatGrad
import random
import warnings
from tqdm import tqdm

__all__ = [
    # Base Class
    "Importance",

    # Basic Group Importance
    "GroupMagnitudeImportance",
    "GroupTaylorImportance",
    "GroupHessianImportance",

    # Aliases
    "MagnitudeImportance",
    "TaylorImportance",
    "HessianImportance",

    # Other Importance
    "BNScaleImportance",
    "LAMPImportance",
    "RandomImportance",
    "MACAwareImportance",
    "VarianceImportance",

    # CNN VBP helpers
    "build_cnn_target_layers",
    "build_cnn_ignored_layers",
]

class Importance(abc.ABC):
    """ Estimate the importance of a tp.Dependency.Group, and return an 1-D per-channel importance score.

        It should accept a group as inputs, and return a 1-D tensor with the same length as the number of channels.
        All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.

        Example:
            ```python
            DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
            group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
            scorer = MagnitudeImportance()    
            imp_score = scorer(group)    
            #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
            min_score = imp_score.min() 
            ``` 
    """
    @abc.abstractclassmethod
    def __call__(self, group: Group) -> torch.Tensor: 
        raise NotImplementedError


class GroupMagnitudeImportance(Importance):
    """ A general implementation of magnitude importance. By default, it calculates the group L2-norm for each channel/dim.
        It supports several variants like:
            - Standard L1-norm of the first layer in a group: MagnitudeImportance(p=1, normalizer=None, group_reduction="first")
            - Group L1-Norm: MagnitudeImportance(p=1, normalizer=None, group_reduction="mean")
            - BN Scaling Factor: MagnitudeImportance(p=1, normalizer=None, group_reduction="mean", target_types=[nn.modules.batchnorm._BatchNorm])

        Args:
            * p (int): the norm degree. Default: 2
            * group_reduction (str): the reduction method for group importance. Default: "mean"
            * normalizer (str): the normalization method for group importance. Default: "mean"
            * target_types (list): the target types for importance calculation. Default: [nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm]

        Example:
    
            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = GroupMagnitudeImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    def __init__(self, 
                 p: int=2, 
                 group_reduction: str="mean", 
                 normalizer: str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.LayerNorm]):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias

    def _lamp(self, scores): # Layer-adaptive Sparsity for the Magnitude-based Pruning
        """
        Normalizing scheme for LAMP.
        """
        # sort scores in an ascending order
        sorted_scores,sorted_idx = scores.view(-1).sort(descending=False)
        # compute cumulative sum
        scores_cumsum_temp = sorted_scores.cumsum(dim=0)
        scores_cumsum = torch.zeros(scores_cumsum_temp.shape,device=scores.device)
        scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
        # normalize by cumulative sum
        sorted_scores /= (scores.sum() - scores_cumsum)
        # tidy up and output
        new_scores = torch.zeros(scores_cumsum.shape,device=scores.device)
        new_scores[sorted_idx] = sorted_scores
        
        return new_scores.view(scores.shape)
    
    def _normalize(self, group_importance, normalizer):
        if normalizer is None:
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        elif normalizer.startswith('sentinel'): # normalize the score with the k-th smallest element. e.g. sentinel_0.5 means median normalization
            sentinel = float(normalizer.split('_')[1]) * len(group_importance)
            sentinel = torch.argsort(group_importance, dim=0, descending=False)[int(sentinel)]
            return group_importance / (group_importance[sentinel]+1e-8)
        elif normalizer=='lamp':
            return self._lamp(group_importance)
        else:
            raise NotImplementedError

    def _reduce(self, group_imp: typing.List[torch.Tensor], group_idxs: typing.List[typing.List[int]]):
        if len(group_imp) == 0: return group_imp
        if self.group_reduction == 'prod':
            reduced_imp = torch.ones_like(group_imp[0], dtype=torch.float32)
        elif self.group_reduction == 'max':
            reduced_imp = torch.ones_like(group_imp[0], dtype=torch.float32) * -99999
        else:
            reduced_imp = torch.zeros_like(group_imp[0], dtype=torch.float32)
        
        n_imp = 0
        for i, (imp, root_idxs) in enumerate(zip(group_imp, group_idxs)):
            imp = imp.to(reduced_imp.device, dtype=reduced_imp.dtype)
            if any([r is None for r in root_idxs]):
                #warnings.warn("Root idxs contain None values. Skipping this layer...")
                continue
            if self.group_reduction == "sum" or self.group_reduction == "mean":
                reduced_imp.scatter_add_(0, torch.tensor(root_idxs, device=imp.device), imp) # accumulated importance
            elif self.group_reduction == "max": # keep the max importance
                selected_imp = torch.index_select(reduced_imp, 0, torch.tensor(root_idxs, device=imp.device))
                selected_imp = torch.maximum(input=selected_imp, other=imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), selected_imp)
            elif self.group_reduction == "prod": # product of importance
                selected_imp = torch.index_select(reduced_imp, 0, torch.tensor(root_idxs, device=imp.device))
                torch.mul(selected_imp, imp, out=selected_imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), selected_imp)
            elif self.group_reduction == 'first':
                if i == 0:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), imp)
            elif self.group_reduction == 'gate':
                if i == len(group_imp)-1:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), imp)
            elif self.group_reduction is None:
                reduced_imp = torch.stack(group_imp, dim=0) # no reduction
            else:
                raise NotImplementedError
            n_imp += 1

        if self.group_reduction == "mean":
            reduced_imp /= n_imp
        return reduced_imp
    
    @torch.no_grad()
    def __call__(self, group: Group):
        group_imp = []
        group_idxs = []

        # Iterate over all groups and estimate group importance
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    local_imp = layer.bias.data[idxs].abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)

                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                
                local_imp = local_imp[idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            ####################
            # BatchNorm
            ####################
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_imp = w.abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            ####################
            # LayerNorm
            ####################
            elif prune_fn == function.prune_layernorm_out_channels:

                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    local_imp = w.abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class BNScaleImportance(GroupMagnitudeImportance):
    """Learning Efficient Convolutional Networks through Network Slimming, 
    https://arxiv.org/abs/1708.06519

    Example:
    
        It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
        All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
        
        ```python
            DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
            group = DG.get_pruning_group( model.bn1, tp.prune_batchnorm_out_channels, idxs=[2, 6, 9] )    
            scorer = BNScaleImportance()    
            imp_score = scorer(group)    
            #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
            min_score = imp_score.min() 
        ``` 

    """

    def __init__(self, group_reduction='mean', normalizer='mean'):
        super().__init__(p=1, group_reduction=group_reduction, normalizer=normalizer, bias=False, target_types=(nn.modules.batchnorm._BatchNorm,))


class LAMPImportance(GroupMagnitudeImportance):
    """Layer-adaptive Sparsity for the Magnitude-based Pruning,
    https://arxiv.org/abs/2010.07611

    Example:
    
            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = LAMPImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """

    def __init__(self, p=2, group_reduction="mean", normalizer='lamp', bias=False):
        assert normalizer == 'lamp'
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer, bias=bias)


class FPGMImportance(GroupMagnitudeImportance):
    """Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration,
    http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.pdf
    """

    def __init__(self, p=2, group_reduction="mean", normalizer='mean', bias=False):
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer, bias=bias)

    @torch.no_grad()
    def __call__(self, group, **kwargs):
        group_imp = []
        group_idxs = []
        # Iterate over all groups and estimate group importance
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_imp = w.abs().pow(self.p)
                # calculate the euclidean distance as similarity
                similar_matrix = torch.cdist(local_imp.unsqueeze(0), local_imp.unsqueeze(0), p=2).squeeze(0)
                similar_sum = torch.sum(torch.abs(similar_matrix), dim=0)
                group_imp.append(similar_sum)
                group_idxs.append(root_idxs)

            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)

                local_imp = w.abs().pow(self.p)

                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                local_imp = local_imp[idxs]
                similar_matrix = torch.cdist(local_imp.unsqueeze(0), local_imp.unsqueeze(0), p=2).squeeze(0)
                similar_sum = torch.sum(torch.abs(similar_matrix), dim=0)
                group_imp.append(similar_sum)
                group_idxs.append(root_idxs)

            # FPGMImportance should not care about BatchNorm and LayerNorm

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class RandomImportance(Importance):
    """ Random importance estimator
    Example:
    
            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = RandomImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    @torch.no_grad()
    def __call__(self, group, **kwargs):
        _, idxs = group[0]
        return torch.rand(len(idxs))


class GroupTaylorImportance(GroupMagnitudeImportance):
    """ Grouped first-order taylor expansion of the loss function.
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf

        Example:

            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                inputs, labels = ...
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                loss = loss_fn(model(inputs), labels)
                loss.backward() # compute gradients
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = GroupTaylorImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 multivariable:bool=False, 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.multivariable = multivariable
        self.target_types = target_types
        self.bias = bias

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue
            
            # Conv/Linear Output
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    dw = layer.weight.grad.data.transpose(1, 0)[
                        idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                    dw = layer.weight.grad.data[idxs].flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    b = layer.bias.data[idxs]
                    db = layer.bias.grad.data[idxs]
                    local_imp = (b * db).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    
            # Conv/Linear Input
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight).flatten(1)
                    dw = (layer.weight.grad).flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)
                    dw = (layer.weight.grad).transpose(0, 1).flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                
                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                local_imp = local_imp[idxs]

                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w*dw).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
            # LN
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w*dw).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class OBDCImportance(GroupMagnitudeImportance):
    """EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis:
       http://proceedings.mlr.press/v97/wang19g/wang19g.pdf
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear],
                 num_classes=100):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self.A, self.DS = {}, {}
        self.Fisher = {}
        self.MatGradHandler = ComputeMatGrad()
        self.steps = 0
        self.eps = 1e-10
        self.modules = []
        self.num_classes = num_classes
        self.known_modules = {'Linear', 'Conv2d'}
    
    def step(self):
        with torch.no_grad():
            for m in self.modules:
                A, DS = self.A[m], self.DS[m]
                grad_mat = self.MatGradHandler(A, DS, m)
                grad_mat *= DS.size(0)
                if self.steps == 0:
                    self.Fisher[m] = grad_mat.new(grad_mat.size()[1:]).fill_(0)
                self.Fisher[m] += (grad_mat.pow_(2)).sum(0)
                self.A[m] = None
                self.DS[m] = None
        self.steps += 1

    def adjust_fisher(self, group, idxs):
        for i, (dep, id) in enumerate(group):
            layer = dep.target.module
            if layer in self.modules:
                if layer.weight.grad is not None:
                    shape = layer.weight.shape
                    if isinstance(layer, nn.modules.conv._ConvNd):
                        kernel_size = shape[2]*shape[3]
                    else:
                        kernel_size = 1
                    indices_to_keep = list(range(self.Fisher[layer].shape[1]))
                    for idx in idxs:
                        indices_to_keep = [i for i in indices_to_keep if not (idx*kernel_size <= i < (idx+1)*kernel_size)]
                    self.Fisher[layer] = torch.index_select(self.Fisher[layer], 1, torch.LongTensor(indices_to_keep).to(self.Fisher[layer].device))
            

    def _rm_hooks(self, model):
        for m in self.modules:
            m._backward_hooks = OrderedDict()
            m._forward_pre_hooks = OrderedDict()

    def _save_input(self, module, input):
        self.A[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        self.DS[module] = grad_output[0].data

    def _prepare_model(self, model, pruner):
        for group in pruner.DG.get_all_groups(ignored_layers=pruner.ignored_layers, root_module_types=pruner.root_module_types): 
            group = pruner._downstream_node_as_root_if_attention(group)
            for i, (dep, idxs) in enumerate(group):
                layer = dep.target.module
                if isinstance(layer, tuple(self.target_types)) and dep.handler in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    self.modules.append(layer)
                    layer.register_forward_pre_hook(self._save_input)
                    layer.register_backward_hook(self._save_grad_output)

    def _clear_buffer(self):
        self.Fisher = {}
        self.modules = []
        self.steps = 0
    
    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)) or (isinstance(layer, torch.nn.Linear) and layer.out_features == self.num_classes):
                continue
            F_diag = (self.Fisher[layer] / self.steps + self.eps)
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    else:
                        w = layer.weight.data[idxs].flatten(1)
                    local_imp = (w ** 2 * F_diag).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                
                if self.bias and layer.bias is not None and layer.bias.grad is not None:
                    b = layer.bias.data[idxs]
                    local_imp = (b ** 2 * F_diag).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class GroupHessianImportance(GroupMagnitudeImportance):
    """Grouped Optimal Brain Damage:
       https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html

       Example:

            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                inputs, labels = ...
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                scorer = GroupHessianImportance()   
                scorer.zero_grad() # clean the acuumulated gradients if necessary
                loss = loss_fn(model(inputs), labels, reduction='none') # compute loss for each sample
                for l in loss:
                    model.zero_grad() # clean the model gradients
                    l.backward(retain_graph=True) # compute gradients for each sample
                    scorer.accumulate_grad(model) # accumulate gradients of each sample
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self._accu_grad = {}
        self._counter = {}

    def zero_grad(self):
        self._accu_grad = {}
        self._counter = {}

    def accumulate_grad(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self._accu_grad:
                    self._accu_grad[param] = param.grad.data.clone().pow(2)
                else:
                    self._accu_grad[param] += param.grad.data.clone().pow(2)
                
                if name not in self._counter:
                    self._counter[param] = 1
                else:
                    self._counter[param] += 1
    
    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []

        if len(self._accu_grad) > 0: # fill gradients so that we can re-use the implementation for Taylor
            for p, g in self._accu_grad.items():
                p.grad.data = g / self._counter[p]
            self.zero_grad()

        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                        h = layer.weight.grad.data.transpose(1, 0)[idxs].flatten(1)
                    else:
                        w = layer.weight.data[idxs].flatten(1)
                        h = layer.weight.grad.data[idxs].flatten(1)

                    local_imp = (w**2 * h).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                
                if self.bias and layer.bias is not None and layer.bias.grad is not None:
                    b = layer.bias.data[idxs]
                    h = layer.bias.grad.data[idxs]
                    local_imp = (b**2 * h)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = (layer.weight).flatten(1)
                        h = (layer.weight.grad).flatten(1)
                    else:
                        w = (layer.weight).transpose(0, 1).flatten(1)
                        h = (layer.weight.grad).transpose(0, 1).flatten(1)

                    local_imp = (w**2 * h).sum(1)
                    if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                        local_imp = local_imp.repeat(layer.groups)
                    local_imp = local_imp[idxs]
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine:
                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs]
                        h = layer.weight.grad.data[idxs]
                        local_imp = (w**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None and layer.bias.grad is None:
                        b = layer.bias.data[idxs]
                        h = layer.bias.grad.data[idxs]
                        local_imp = (b**2 * h).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
            # LN
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs]
                        h = layer.weight.grad.data[idxs]
                        local_imp = (w**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None and layer.bias.grad is not None:
                        b = layer.bias.data[idxs]
                        h = layer.bias.grad.data[idxs]
                        local_imp = (b**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


# Aliases
class MagnitudeImportance(GroupMagnitudeImportance):
    pass

class TaylorImportance(GroupTaylorImportance):
    pass

class HessianImportance(GroupHessianImportance):
    pass

from contextlib import contextmanager

class ActivationImportance(GroupMagnitudeImportance):

    @contextmanager
    def compute_importance(self, model):
        
        @torch.no_grad()
        def _compute_importance_hook(module, input, output):

            if isinstance(module, nn.Linear):
                dim = input[0].shape[-1]
                module._importance = input[0].abs().view(-1, dim).sum(0)
            elif isinstance(module, nn.Conv2d):
                dim = input[0].shape[1]
                module._importance = input[0].abs().mean((0, 2, 3))
            return 
        
        hooks = []
        for m in model.modules():
            if isinstance(m, tuple(self.target_types)):
                hooks.append(m.register_forward_hook(_compute_importance_hook))
        
        yield

        for h in hooks:
            h.remove()

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue
            
            # Conv/Linear Output
            if prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if not hasattr(layer, "_importance"):
                    warnings.warn("Layer {} does not have _importance attribute.".format(layer))
                    continue
                local_imp = layer._importance[idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class MACAwareImportance(GroupMagnitudeImportance):
    """MAC-aware importance that combines base importance with layer MAC costs.

    This importance criterion scales the base importance score by considering
    the computational cost (MACs) of each layer, encouraging pruning of
    channels in computationally expensive layers.

    Args:
        p (int): Norm degree for base importance calculation. Default: 2
        layers_mac (dict): Dictionary mapping layer names to their MAC counts.
            Required parameter.
        params (dict): Configuration parameters with keys:
            - "type": Combination method ("Sum" or "Mul"). Default: "Sum"
            - "alpha": Weight for importance vs MAC trade-off (0-1). Default: 0.9
            - "beta": Exponent for MAC ratio calculation. Default: 2
            - "use_macs": Whether to use MAC-aware scoring. Default: True
        current_max (float): Current maximum importance for normalization in "Sum" mode.
            Required when type="Sum".

    Example:
        ```python
        # First compute per-layer MACs using torch_pruning utilities
        layers_mac = {"conv1": 1000000, "conv2": 2000000, ...}

        scorer = MACAwareImportance(
            layers_mac=layers_mac,
            params={"type": "Sum", "alpha": 0.9, "beta": 2, "use_macs": True},
            current_max=1.0
        )

        DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224))
        group = DG.get_pruning_group(model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9])
        imp_score = scorer(group)
        ```
    """

    def __init__(self, p=2, layers_mac=None, params=None, current_max=None):
        super().__init__(p=p)
        if params is None:
            params = {"type": "Sum", "alpha": 0.9, "beta": 2, "use_macs": True}
        self.use_macs = params.get("use_macs", True)
        assert layers_mac is not None, "layers_mac must be provided"
        self.layers_mac = layers_mac
        self.min_layer_mac = min(self.layers_mac.values())
        self.max_layer_mac = max(self.layers_mac.values())
        self.combination_type = params.get("type", "Sum")
        self.alpha = params.get("alpha", 0.9)
        self.beta = params.get("beta", 2)
        self.current_max = current_max

    @torch.no_grad()
    def __call__(self, group, act_only=False):
        """Compute MAC-aware importance for a pruning group.

        Args:
            group: Pruning group from DependencyGraph
            act_only: If True, returns base importance without MAC scaling
                (for computing max importance normalization)

        Returns:
            1-D tensor of importance scores per channel
        """
        # Compute base importance using parent class
        base_importance = super().__call__(group)

        if base_importance is None:
            return None

        if not self.use_macs or act_only:
            return base_importance

        # Extract layer name from group
        dep, _ = group[0]
        layer_name = dep.target.name
        # Handle format "layer_name (Module)" by extracting just the name
        if " " in layer_name:
            layer_name = layer_name[:layer_name.index(" ")]
        # Strip "module." prefix if present (for DataParallel models)
        if layer_name.startswith("module."):
            layer_name = layer_name[len("module."):]

        # Get MAC cost for this layer
        if layer_name not in self.layers_mac:
            warnings.warn(f"Layer {layer_name} not found in layers_mac, returning base importance")
            return base_importance

        layer_mac = self.layers_mac[layer_name]

        # Normalize MAC to [0, 1] range
        mac_range = self.max_layer_mac - self.min_layer_mac
        if mac_range > 0:
            layer_mac_norm = (layer_mac - self.min_layer_mac) / mac_range
        else:
            layer_mac_norm = 0.5

        # Compute MAC ratio for multiplicative scaling
        ratio_mac_min = (self.min_layer_mac / layer_mac) ** (1 / self.beta)

        # Combine importance with MAC cost
        if self.combination_type == "Mul":
            # Multiplicative: scale importance by MAC ratio
            # Channels in high-MAC layers get lower effective importance
            combined_importance = base_importance * ratio_mac_min
        elif self.combination_type == "Sum":
            # Additive: weighted sum of normalized importance and MAC penalty
            # Higher MAC layers get bonus to encourage pruning there
            # Handle current_max being None, scalar 0, or tensor with value 0
            use_local_max = (
                self.current_max is None or
                (isinstance(self.current_max, (int, float)) and self.current_max == 0) or
                (torch.is_tensor(self.current_max) and self.current_max.item() == 0)
            )
            if use_local_max:
                norm_importance = base_importance / (base_importance.max() + 1e-8)
            else:
                norm_importance = base_importance / self.current_max
            mac_penalty = 1 - layer_mac_norm  # Higher MAC = lower penalty = lower importance
            combined_importance = self.alpha * norm_importance + (1 - self.alpha) * mac_penalty
        else:
            combined_importance = base_importance

        return combined_importance


class VarianceImportance(Importance):
    """
    Variance-based activation importance (Exact VBP).

    Usage:
        importance = VarianceImportance(norm_per_layer=False)
        importance.collect_statistics(model, train_loader, device)
        pruner = MagnitudePruner(..., importance=importance)
    """

    def __init__(self, norm_per_layer: bool = False, eps: float = 1e-8):
        super().__init__()
        self.norm_per_layer = norm_per_layer
        self.eps = eps

        # Exact accumulators: module -> tensors
        self.sum = {}
        self.sum_sq = {}
        self.count = {}

        # final per-module statistics after collection
        self.variance = {}   # module -> var[C]
        self.means = {}      # module -> mean[C]

    # ---------------------------------------------------------
    # 1. Collect statistics OFFLINE (NO EMA)
    # ---------------------------------------------------------
    @torch.no_grad()
    def collect_statistics(self, model, train_loader, device, target_layers=None, max_batches=200):
        """
        Collect per-channel activation statistics.

        Args:
            model: The model to collect statistics from.
            train_loader: DataLoader for training/calibration data.
            device: Device to run on.
            target_layers: Optional list of (module, post_act_fn) tuples.
                If provided, only hooks these modules and applies post_act_fn
                to the output before collecting stats. This is required for
                architectures like ViT where VBP needs post-GELU activation
                stats on MLP fc1 layers specifically.
                If None, hooks all Conv2d/Linear with no post-activation.
            max_batches: Maximum number of batches to collect statistics for.
        """
        self.sum.clear()
        self.sum_sq.clear()
        self.count.clear()
        self.variance.clear()
        self.means.clear()

        handles = []
        if target_layers is not None:
            for module, post_act_fn in target_layers:
                handles.append(
                    module.register_forward_hook(
                        self._make_conv_linear_hook(module, post_act_fn=post_act_fn)
                    )
                )
        else:
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    handles.append(m.register_forward_hook(self._make_conv_linear_hook(m)))

        model.eval()
        total = min(len(train_loader), max_batches) if max_batches else len(train_loader)
        pbar = tqdm(train_loader, desc="Collecting stats", total=total, miniters=max(total // 20, 1))
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            model(images)
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

        for h in handles:
            h.remove()

        self._compute_statistics()

    # ---------------------------------------------------------
    # Hook: accumulate sum(x), sum(x^2), count
    # ---------------------------------------------------------
    def _make_conv_linear_hook(self, module, post_act_fn=None):
        def hook(mod, inp, out):
            if out is None:
                return
            x = out.detach()

            # Apply post-activation (e.g., GELU for ViT MLP fc1)
            if post_act_fn is not None:
                x = post_act_fn(x)

            if x.dim() == 4:
                # Conv2d: [B, C, H, W] -> [B*H*W, C]
                x = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
            elif x.dim() == 3:
                # Transformer Linear: [B, T, C] -> [B*T, C]
                x = x.reshape(-1, x.size(-1))
            elif x.dim() == 2:
                # Standard Linear: [B, C]
                pass
            else:
                return

            n = x.size(0)
            if mod not in self.sum:
                self.sum[mod] = x.sum(dim=0).cpu()
                self.sum_sq[mod] = (x * x).sum(dim=0).cpu()
                self.count[mod] = n
            else:
                self.sum[mod] += x.sum(dim=0).cpu()
                self.sum_sq[mod] += (x * x).sum(dim=0).cpu()
                self.count[mod] += n

        return hook

    # ---------------------------------------------------------
    # Compute final mean and variance: var = E[x^2] - (E[x])^2
    # ---------------------------------------------------------
    def _compute_statistics(self):
        self.variance.clear()
        self.means.clear()

        for module in self.sum.keys():
            s = self.sum[module]
            s2 = self.sum_sq[module]
            n = float(self.count[module])

            mean = s / n               # [C]
            mean_sq = s2 / n           # [C]
            var = mean_sq - mean * mean

            # Numerical safety
            var = torch.clamp(var, min=0.0)

            # Optional normalization per layer
            if self.norm_per_layer:
                layer_mean = var.mean()
                if layer_mean > 0:
                    var = var / (layer_mean + self.eps)

            self.means[module] = mean.clone()   # <-- for VBP compensation
            self.variance[module] = var         # <-- for importance

    # ---------------------------------------------------------
    # 2. Torch-Pruning interface: importance(group)
    # ---------------------------------------------------------
    @torch.no_grad()
    def __call__(self, group, **kwargs):
        """
        Returns importance values for the channels in this pruning group.
        Lower variance => lower importance => pruned first by MagnitudePruner.
        """
        dep, idxs = group[0]
        module = dep.target.module

        if module not in self.variance:
            # If this module had no stats (unlikely), fallback
            return torch.ones(len(idxs))

        var = self.variance[module]
        idxs = torch.as_tensor(idxs, dtype=torch.long)
        scores = var[idxs].clone()

        return scores


# ---------------------------------------------------------------------------
# CNN VBP helpers: auto-detect target_layers and ignored_layers
# ---------------------------------------------------------------------------

def _grad_fn_to_activation(grad_fn_str):
    """Map a grad_fn name to a functional activation, or None."""
    import torch.nn.functional as _F
    _mapping = {
        "relu": _F.relu,
        "silu": _F.silu,
        "gelu": _F.gelu,
        "relu6": _F.relu6,
        "hardswish": _F.hardswish,
        "hardsigmoid": _F.hardsigmoid,
    }
    grad_fn_lower = grad_fn_str.lower()
    for key, fn in _mapping.items():
        if key in grad_fn_lower:
            return fn
    return None


def _compose_post_act(conv_node):
    """Walk forward from conv node to find BN + activation, return composed callable or None.

    Returns:
        (post_act_fn, description_str) or (None, None)
    """
    from ..ops import _ElementWiseOp

    bn_module = None
    act_fn = None

    # Walk immediate outputs of conv node
    for out_node in conv_node.outputs:
        mod = out_node.module
        # Check for BatchNorm
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            bn_module = mod
            # Walk further from BN to find activation
            for bn_out in out_node.outputs:
                bn_out_mod = bn_out.module
                if isinstance(bn_out_mod, _ElementWiseOp):
                    act_fn = _grad_fn_to_activation(bn_out_mod._grad_fn)
                    if act_fn is not None:
                        break
            break
        # Check for direct activation (no BN)
        if isinstance(mod, _ElementWiseOp):
            act_fn = _grad_fn_to_activation(mod._grad_fn)
            if act_fn is not None:
                break

    # Compose
    if bn_module is not None and act_fn is not None:
        def _composed(x, _bn=bn_module, _act=act_fn):
            return _act(_bn(x))
        return _composed
    elif bn_module is not None:
        return bn_module
    elif act_fn is not None:
        return act_fn
    return None


def build_cnn_target_layers(model, DG):
    """Auto-detect target_layers for CNN VBP stats collection.

    Walks DG forward from each Conv2d to find BN + activation,
    composes them into post_act_fn for correct mean collection.

    Args:
        model: CNN model (e.g. torchvision ResNet, MobileNetV2)
        DG: Built DependencyGraph with module2node populated

    Returns:
        list of (conv_module, post_act_fn) tuples
    """
    target_layers = []
    for module, node in DG.module2node.items():
        if not isinstance(module, nn.Conv2d):
            continue
        # Skip depthwise convolutions
        if module.groups == module.out_channels and module.out_channels > 1:
            continue
        # Skip 1-channel convs (stems with groups=1 are fine, but 1-output is degenerate)
        if module.out_channels == 1:
            continue

        post_act_fn = _compose_post_act(node)
        target_layers.append((module, post_act_fn))

    return target_layers


def _is_bottleneck_resnet(model):
    """Check if model uses Bottleneck blocks (ResNet-50/101/152)."""
    try:
        from torchvision.models.resnet import Bottleneck
        return any(isinstance(m, Bottleneck) for m in model.modules())
    except ImportError:
        return False


def _find_classifier(model):
    """Find the classifier layer in a torchvision model."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc
    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Linear):
            return clf
        # MobileNetV2 classifier is nn.Sequential
        if isinstance(clf, nn.Sequential):
            return clf
    return None


def build_cnn_ignored_layers(model, architecture, interior_only=True):
    """Build ignored_layers list for CNN VBP pruning.

    Args:
        model: torchvision model
        architecture: e.g. "resnet50", "resnet18", "mobilenet_v2"
        interior_only: If True, only prune interior block channels
            (conv1/conv2 in Bottleneck, expand conv in MobileNetV2).
            If False, prune all convs except stem+classifier.

    Returns:
        list of modules to ignore during pruning
    """
    ignored = []

    # Always ignore classifier
    clf = _find_classifier(model)
    if clf is not None:
        if isinstance(clf, nn.Sequential):
            for m in clf.modules():
                ignored.append(m)
        else:
            ignored.append(clf)

    if architecture.startswith("resnet"):
        # Stem conv
        if hasattr(model, "conv1"):
            ignored.append(model.conv1)

        is_bottleneck = _is_bottleneck_resnet(model)

        if interior_only and is_bottleneck:
            # Bottleneck: ignore conv3 (output to residual stream) + all downsamples
            from torchvision.models.resnet import Bottleneck
            for m in model.modules():
                if isinstance(m, Bottleneck):
                    ignored.append(m.conv3)
                    if m.downsample is not None:
                        for sub in m.downsample.modules():
                            if isinstance(sub, nn.Conv2d):
                                ignored.append(sub)
        elif interior_only and not is_bottleneck:
            # BasicBlock: no true interior — same as full pruning but with downsample ignored
            from torchvision.models.resnet import BasicBlock
            for m in model.modules():
                if isinstance(m, BasicBlock):
                    if m.downsample is not None:
                        for sub in m.downsample.modules():
                            if isinstance(sub, nn.Conv2d):
                                ignored.append(sub)
        else:
            # interior_only=False: only ignore stem+classifier+downsample convs
            for name, m in model.named_modules():
                if "downsample" in name and isinstance(m, nn.Conv2d):
                    ignored.append(m)

    elif architecture == "mobilenet_v2":
        # Stem conv (first conv in features[0])
        if hasattr(model, "features"):
            for m in model.features[0].modules():
                if isinstance(m, nn.Conv2d):
                    ignored.append(m)

        if interior_only:
            # Ignore: projection convs (1x1 reduce) and DW convs
            # In InvertedResidual, the conv sequence is:
            #   expand 1x1 -> DW 3x3 -> project 1x1
            # We want to keep only the expand conv
            try:
                from torchvision.models.mobilenetv2 import InvertedResidual
            except ImportError:
                from torchvision.models.mobilenet import InvertedResidual

            for m in model.modules():
                if isinstance(m, InvertedResidual):
                    convs = [sub for sub in m.conv.modules() if isinstance(sub, nn.Conv2d)]
                    for conv in convs:
                        # DW conv: groups == out_channels
                        if conv.groups == conv.out_channels and conv.out_channels > 1:
                            ignored.append(conv)
                        # Projection conv: last 1x1 (pointwise, groups=1, not the expand)
                        elif conv.kernel_size == (1, 1) and conv == convs[-1]:
                            ignored.append(conv)
        else:
            # Only ignore DW convs (can't resize them independently)
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    if m.groups == m.out_channels and m.out_channels > 1:
                        ignored.append(m)

    return ignored