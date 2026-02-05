import torch
import torch.nn as nn


def load_state_dict_pruned(model, state_dict):
    """
    Loads a state_dict into the model, adjusting layers with mismatched shapes due to pruning.

    Args:
        model (nn.Module): The original model.
        state_dict (dict): The state_dict with pruned weights.

    Returns:
        nn.Module: The model with adjusted layers and loaded weights.
    """
    model_state_dict = model.state_dict()
    mismatched_layers = []

    # Identify mismatched layers
    for key in state_dict.keys():
        if key in model_state_dict:
            if state_dict[key].shape != model_state_dict[key].shape:
                print(f"Mismatch found in '{key}': model shape {model_state_dict[key].shape}, checkpoint shape {state_dict[key].shape}")
                mismatched_layers.append(key)
        else:
            print(f"Key '{key}' not found in the model's state_dict.")

    # Adjust layers in the model to match the pruned weights
    for key in mismatched_layers:
        module_name = '.'.join(key.split('.')[:-1])  # Get the module name
        param_type = key.split('.')[-1]             # 'weight' or 'bias'

        module = get_module_by_name(model, module_name)
        if module is None:
            print(f"Module '{module_name}' not found in the model.")
            continue

        new_param = state_dict[key]

        if isinstance(module, nn.Conv2d):
            adjust_conv2d(model, module_name, module, new_param, param_type)
            print(f"Adjusted Conv2d layer '{module_name}.{param_type}' to match pruned weights.")
        elif isinstance(module, nn.ConvTranspose2d):
            adjust_convtranspose2d(model, module_name, module, new_param, param_type)
            print(f"Adjusted ConvTranspose2d layer '{module_name}.{param_type}' to match pruned weights.")
        elif isinstance(module, nn.Linear):
            adjust_linear(model, module_name, module, new_param, param_type)
            print(f"Adjusted Linear layer '{module_name}.{param_type}' to match pruned weights.")
        elif isinstance(module, nn.BatchNorm2d):
            adjust_batchnorm2d(model, module_name, module, new_param)
            print(f"Adjusted BatchNorm2d layer '{module_name}.{param_type}' to match pruned weights.")
        elif isinstance(module, nn.PReLU):
            adjust_prelu(model, module_name, module, new_param)
            print(f"Adjusted PReLU layer '{module_name}' to match pruned weights.")
        else:
            print(f"Layer type '{type(module)}' for '{module_name}' is not supported for adjustment.")

    # Load the state_dict with strict=False to allow loading into adjusted layers
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Successfully loaded pruned state_dict into the model.")
    return model, missing_keys, unexpected_keys


def get_module_by_name(model, module_name):
    """
    Retrieves a module from the model based on its dotted module name.

    Args:
        model (nn.Module): The model containing the module.
        module_name (str): The dotted module name (e.g., 'layer1.conv1').

    Returns:
        nn.Module or None: The retrieved module or None if not found.
    """
    names = module_name.split('.')
    module = model
    for name in names:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            return None
    return module


def adjust_conv2d(model, module_name, module, new_param, param_type):
    """
    Adjusts a Conv2d module to match the shape of the new parameter.

    Args:
        model (nn.Module): The model containing the module.
        module_name (str): The name of the module.
        module (nn.Conv2d): The original Conv2d module.
        new_param (torch.Tensor): The new parameter tensor.
        param_type (str): Type of the parameter ('weight' or 'bias').
    """
    old_module = module
    new_out_channels = new_param.shape[0]
    new_groups = 1 if old_module.groups == 1 else new_out_channels
    # if param_type == 'weight':
    #     new_in_channels = new_param.shape[1] * new_groups
    # else:
    #     new_in_channels = old_module.in_channels

    if param_type == 'weight':
        is_depthwise = old_module.groups == old_module.in_channels == old_module.out_channels
        if is_depthwise:
            new_in_channels = new_param.shape[0]  # Match out_channels
            new_groups = new_param.shape[0]  # Depthwise: groups = in = out
        else:
            new_in_channels = new_param.shape[1]
            new_groups = old_module.groups  # has to be checked!
    else:
        new_in_channels = old_module.in_channels
        new_groups = old_module.groups

    new_module = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=new_out_channels,
        kernel_size=old_module.kernel_size,
        stride=old_module.stride,
        padding=old_module.padding,
        dilation=old_module.dilation,
        groups=new_groups,
        bias=old_module.bias is not None,
        padding_mode=old_module.padding_mode
    )

    # Replace the module in the model
    set_module_by_name(model, module_name, new_module)


def adjust_convtranspose2d(model, module_name, module, new_param, param_type):
    """
    Adjusts a ConvTranspose2d module to match the shape of the new parameter.

    Args:
        model (nn.Module): The model containing the module.
        module_name (str): The name of the module.
        module (nn.ConvTranspose2d): The original ConvTranspose2d module.
        new_param (torch.Tensor): The new parameter tensor.
        param_type (str): Type of the parameter ('weight' or 'bias').
    """
    old_module = module
    if param_type == 'weight':
        new_in_channels = new_param.shape[0]
        new_out_channels = new_param.shape[1]
    else:
        new_in_channels = old_module.in_channels
        new_out_channels = new_param.shape[0]

    new_module = nn.ConvTranspose2d(
        in_channels=new_in_channels,
        out_channels=new_out_channels,
        kernel_size=old_module.kernel_size,
        stride=old_module.stride,
        padding=old_module.padding,
        output_padding=old_module.output_padding,
        groups=old_module.groups,
        bias=old_module.bias is not None,
        dilation=old_module.dilation,
        padding_mode=old_module.padding_mode
    )

    # Replace the module in the model
    set_module_by_name(model, module_name, new_module)


def adjust_linear(model, module_name, module, new_param, param_type):
    """
    Adjusts a Linear module to match the shape of the new parameter.

    Args:
        model (nn.Module): The model containing the module.
        module_name (str): The name of the module.
        module (nn.Linear): The original Linear module.
        new_param (torch.Tensor): The new parameter tensor.
        param_type (str): Type of the parameter ('weight' or 'bias').
    """
    old_module = module
    if param_type == 'weight':
        new_out_features = new_param.shape[0]
        new_in_features = new_param.shape[1]
    else:
        new_out_features = new_param.shape[0]
        new_in_features = old_module.in_features

    new_module = nn.Linear(
        in_features=new_in_features,
        out_features=new_out_features,
        bias=old_module.bias is not None
    )

    # Replace the module in the model
    set_module_by_name(model, module_name, new_module)


def adjust_batchnorm2d(model, module_name, module, new_param):
    """
    Adjusts a BatchNorm2d module to match the shape of the new parameter.

    Args:
        model (nn.Module): The model containing the module.
        module_name (str): The name of the module.
        module (nn.BatchNorm2d): The original BatchNorm2d module.
        new_param (torch.Tensor): The new parameter tensor.
    """
    old_module = module
    new_num_features = new_param.shape[0]

    new_module = nn.BatchNorm2d(
        num_features=new_num_features,
        eps=old_module.eps,
        momentum=old_module.momentum,
        affine=old_module.affine,
        track_running_stats=old_module.track_running_stats
    )

    # Replace the module in the model
    set_module_by_name(model, module_name, new_module)


def adjust_prelu(model, module_name, module, new_param):
    """
    Adjusts a PReLU module to match the shape of the new parameter.

    Args:
        model (nn.Module): The model containing the module.
        module_name (str): The name of the module.
        module (nn.PReLU): The original PReLU module.
        new_param (torch.Tensor): The new parameter tensor.
    """
    old_module = module
    num_parameters = new_param.shape[0]

    new_module = nn.PReLU(
        num_parameters=num_parameters,
        init=old_module.weight.data.mean().item()
    )

    # Replace the module in the model
    set_module_by_name(model, module_name, new_module)


def set_module_by_name(model, module_name, new_module):
    """
    Sets a module in the model based on its dotted module name.

    Args:
        model (nn.Module): The model containing the module.
        module_name (str): The dotted module name (e.g., 'layer1.conv1').
        new_module (nn.Module): The new module to set.
    """
    names = module_name.split('.')
    module = model
    for name in names[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            print(f"Module '{'.'.join(names[:-1])}' not found when setting '{module_name}'.")
            return
    setattr(module, names[-1], new_module)
