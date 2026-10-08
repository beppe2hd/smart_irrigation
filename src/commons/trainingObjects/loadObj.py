import torch
from torch import nn

available_optimizers = ["SGD", "Adam"]
available_losses = ["MSELoss", "Huber", "MSE_W"]

import torch
import torch.nn as nn

class ExpWeightedMSELoss(nn.Module):
    #def __init__(self, horizon=48, decay=0.1):
    def __init__(self, device = torch.device("cpu"), horizon=48, decay=0.1):
        super().__init__()
        t = torch.arange(horizon).float()
        weights = torch.exp(-decay * t)


        # Normalizzazione opzionale (mantiene scala simile alla MSE)
        weights = weights / weights.sum() * horizon
        self.device = device

        self.register_buffer("weights", weights)  # non è un parametro

    def forward(self, pred, target):
        # pred, target: (B, 48, 2)

        loss = (pred - target) ** 2   # (B, 48, 2)

        # reshape pesi per broadcast su feature
        w = self.weights.view(1, -1, 1).to(self.device)  # (1, 48, 1)

        weighted_loss = loss * w

        return weighted_loss.mean()


def load_Optimizer(model, config):

    oprimizer_type = config["hyperparameters"]["optimizer"]
    lr = config["hyperparameters"]["learning_rate"]
    momentum = config["hyperparameters"]["momentum"]
    weight_decay = config["hyperparameters"]["weight_decay"]

    if oprimizer_type in available_optimizers:

        if oprimizer_type == "SGD":

            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
    
        if oprimizer_type == "Adam":

            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

        return optimizer

    else:
        print(
            f"Optimizer {oprimizer_type}, not available, chose one in {available_optimizers}"
        )


def load_Loss(config, device):

    loss_type = config["hyperparameters"]["loss"]

    if loss_type in available_losses:

        if loss_type == "MSELoss":
            criterion = nn.MSELoss()

        if loss_type == "MAPE":
            def mape(y_pred, y_true, epsilon=1e-8):
                return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
            criterion = mape

        if loss_type == "MSE_W":
            criterion = ExpWeightedMSELoss(decay=config["hyperparameters"]["loss_MSE_W_decay"], device = device)
            #criterion = ExpWeightedMSELoss(decay=config["hyperparameters"]["loss_MSE_W_decay"])
            #print(device)

        if loss_type == "Huber":
            criterion = nn.HuberLoss(delta=1.0)

    return criterion
