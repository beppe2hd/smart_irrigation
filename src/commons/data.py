import torch

def inverse_scale_data(output, cols, scaler, device):

    mean = torch.tensor(scaler.mean_[cols])
    std = torch.tensor(scaler.scale_[cols])

    std = std.view(1, 1, len(cols)).to(device=device, dtype=torch.float32)  # → shape [1, 2, 1]
    mean = mean.view(1, 1, len(cols)).to(device=device, dtype=torch.float32)


    output = output * std + mean

    return output


def specific_scale_data(input, cols, scaler, device):

    mean = torch.tensor(scaler.mean_[cols])
    std = torch.tensor(scaler.scale_[cols])

    std = std.view(1, 1, len(cols)).to(device=device, dtype=torch.float32)  # → shape [1, 2, 1]
    mean = mean.view(1, 1, len(cols)).to(device=device, dtype=torch.float32)

    input = (input - mean) / std

    return input