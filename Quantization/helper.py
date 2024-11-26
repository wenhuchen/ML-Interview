import torch
import torch.nn as nn
import numpy as np
import copy

@torch.no_grad()
def compute_activation_stats(model, calibration_loader):
    """
    Compute activation statistics for each layer using calibration data
    """
    input_feats = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            input = input[0]
            if name not in input_feats:
                input_feats[name] = []
            input_feats[name].append(input.detach())
        return hook
    
    # Register hooks for linear layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run calibration data through model
    for batch in calibration_loader:
        X = batch[0].to('cuda')
        model(X)

    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute statistics
    stats = {}
    for name, feat in input_feats.items():
        stats[name] = torch.cat(feat, dim=0)
    
    return stats

q_config = {
    "zero_point": True,  # by default True
    "q_group_size": -1,  # whether to use group quantization
}

def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)

@torch.no_grad()
def search_module_scale(layer, x, w_bit: int = 8):
    # w: co, ci
    # x: n, ci

    x = x.to(layer.weight.device)
    with torch.no_grad():
        org_out = layer(x)
        if isinstance(org_out, tuple):
            org_out = org_out[0]

    x_max = get_act_scale(x)

    best_error = float("inf")
    best_ratio = -1
    best_scales = None

    n_grid = 20
    history = []

    org_sd = copy.deepcopy(layer.weight.data.cpu())

    for ratio in range(n_grid):
        ratio = ratio * 1 / n_grid
        scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()

        # Reconstruct the weight
        layer.weight.mul_(scales.view(1, -1).to(layer.weight.device))
        quantized = pseudo_quantize_tensor(layer.weight.data, n_bit=w_bit, **q_config)
        quantized = quantized.detach()
        layer.weight.data = quantized / (scales.view(1, -1))

        # Propagate the outputs
        out = layer(x)
        loss = (
            (org_out - out).float().pow(2).mean().item()
        )
        history.append(loss)

        # Selecting the best scale
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_scales = scales
        layer.weight.data = org_sd.to('cuda')
    
    assert torch.isnan(best_scales).sum() == 0, best_scales
    
    best_scales = best_scales.detach().cpu()

    quantized_layer = nn.Linear(layer.in_features, layer.out_features, bias=False)
    quantized_layer.weight.data = pseudo_quantize_tensor(
        org_sd, n_bit=w_bit, **q_config) / (best_scales.view(1, -1))

    return quantized_layer, best_scales