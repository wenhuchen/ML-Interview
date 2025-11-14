import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from helper import *

class SimpleNN(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[128, 256, 128, 10]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=False))
            if hidden_dim != hidden_dims[-1]:
                layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class QuantizedNN(nn.Module):
    def __init__(self, quantized_state):
        super().__init__()
        self.layers = nn.ModuleList(quantized_state['layers'])
        self.scales = quantized_state['scales']

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def create_sample_data(num_samples=1000, input_dim=64, num_classes=10):
    """Create synthetic data for testing"""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def quantize_model(model, calibration_loader, bits=4):
    """Quantize the entire model using AWQ"""
    # Compute activation statistics
    input_feats = compute_activation_stats(model, calibration_loader)

    quantized_layers = []
    scales_dict = {}

    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            layer_name = f'layers.{i}'
            q_layer, q_info = search_module_scale(
                layer,
                input_feats[layer_name],
                w_bit=bits
            )
            quantized_layers.append(q_layer)
            scales_dict[layer_name] = q_info
        else:
            quantized_layers.append(layer)

    quantized_state = {
        'layers': quantized_layers,
        'scales': scales_dict
    }

    return QuantizedNN(quantized_state)


def main():
    # Parameters
    input_dim = 64
    hidden_dims = [128, 256, 128, 10]
    batch_size = 32
    quantization_bits = 8

    # Create synthetic data
    X_train, y_train = create_sample_data(1000, input_dim, hidden_dims[-1])

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create and train original model
    model = SimpleNN(input_dim, hidden_dims).to('cuda')

    print('Original weight...')
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            print(f'layers.{i}-------------------------------------')
            print(layer.weight)

    # Quantize model
    quantized_model = quantize_model(model, train_loader, quantization_bits).to('cuda')

    print('\n\nQuantized weight...')
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            print(f'layers.{i}-------------------------------------')
            print(layer.weight)

    print("\nComputing weight error")
    # Display the original and quantized models
    diff = []
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            diff.append(layer.weight.data)

    for layer in quantized_model.layers:
        if isinstance(layer, nn.Linear):
            prior_weight = diff.pop(0)
            cur_weight = layer.weight.data
            accumulate = abs(prior_weight - cur_weight).sum()
            diff.append(accumulate.cpu().numpy().item())

    print('Error Accumulation: ', diff)

    # Evaluate quantized model
    with torch.no_grad():
        y = model(torch.ones((64,)).to('cuda'))
        print('Pre-quantization:\n', y.cpu().numpy())

        y = quantized_model(torch.ones((64,)).to('cuda'))
        print('Pre-quantization:\n', y.cpu().numpy())

    # Calculate model size reduction
    original_size = sum(p.numel() * 32 for p in model.parameters()) / 8  # in bytes
    quantized_size = sum(p.numel() * quantization_bits for p in quantized_model.parameters()) / 8  # in bytes
    size_reduction = (original_size - quantized_size) / original_size * 100

    print(f"\nModel size reduction: {size_reduction:.2f}%")

if __name__ == "__main__":
    main()