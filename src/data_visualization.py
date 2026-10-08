#%%
import joblib, sys, os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))
print(os.getenv("PYTHONPATH"))
from src.commons.utils import get_config_file

config = get_config_file("../src/configurations/config_season2_MSE_W_allFIleds16.yaml")

folder_path = "../src/weights/" + config["name"] + config["version"].replace(".", "_")
path_test_output = folder_path + "/test_output.pkl"
path_test_y = folder_path + "/test_y.pkl"

y_pred = joblib.load(path_test_output)
y_true = joblib.load(path_test_y)

print(config['name'])



# %%
# Plot mean of absolute error
abs_error = abs(y_pred-y_true)
meanError = abs_error.mean(dim=0)
meanError = meanError.detach().cpu().numpy()

plt.plot(meanError)
plt.title("SM abs Error (mean)"); plt.xlabel("time forecast Horizon"); plt.ylabel("variance")
plt.grid(True, alpha=0.3)
plt.show()

#%%
import numpy as np
qs = [0, 10, 25, 50, 75, 90, 100]
err_q = np.percentile(abs_error.detach().cpu().numpy(), qs, axis=0)
err_q = err_q.squeeze()
t = np.arange(err_q.shape[1])

plt.plot(t, err_q[3], label='p50')
plt.fill_between(t, err_q[2], err_q[4], alpha=0.3, label='p25-p75')
plt.fill_between(t, err_q[1], err_q[5], alpha=0.2, label='p10-p90')
#plt.plot(t, err_q[0], '--', alpha=0.5, label='min')
#plt.plot(t, err_q[6], '--', alpha=0.5, label='max')
plt.legend()
plt.xlabel("Timestep")
plt.ylabel("Absolute error")
plt.show()



# %%
# Plot absolute differenze betweehn each time stamp and the first one in the gt sequences
y_diff = abs(y_true-y_true[:,0,:].unsqueeze(1))
y_diff = y_diff.mean(dim=0)
plt.plot(y_diff)
plt.title("Absolute differenze against the start on GT"); plt.xlabel("time forecast Horizon"); plt.ylabel("difference")
plt.grid(True, alpha=0.3)
plt.show()

#%%
err = (y_pred - y_true).squeeze(-1)          # [B, T]
mae_t  = err.abs().mean(dim=0)               # [T]
rmse_t = (err.pow(2).mean(dim=0)).sqrt()     # [T]
plt.plot(mae_t)
# %%
y_diff = abs(y_true[:,0,:]-y_true[:,-1,:])
print(y_diff.mean(dim=0))


# %%
import torch

def exponential_decay_offset_batch(
    last_obs: torch.Tensor,        # shape (B,)
    forecast: torch.Tensor,        # shape (B, H)
    decay_lambda: float = 0.3
) -> torch.Tensor:

    last_obs = last_obs.float()
    forecast = forecast.float()

    delta = last_obs - forecast[:, 0]          # (B,)

    B, H = forecast.shape
    t = torch.arange(1, H + 1, device=forecast.device, dtype=forecast.dtype)
    decay = torch.exp(-decay_lambda * t)       # (H,)

    adjusted = forecast + delta.unsqueeze(1) * decay.unsqueeze(0)

    return adjusted

def linear_blend_bridge_batch(
    last_obs: torch.Tensor,     # shape (B,)
    forecast: torch.Tensor,     # shape (B, H)
    transition_steps: int = 15
) -> torch.Tensor:
    """
    Smoothly connects last observation with forecast
    using linear blending over first k steps.

    Args:
        last_obs: Tensor of shape (B,)
        forecast: Tensor of shape (B, H)
        transition_steps: number of smoothing steps (k)

    Returns:
        Adjusted forecast tensor of shape (B, H)
    """

    last_obs = last_obs.float()
    forecast = forecast.float()

    B, H = forecast.shape
    k = min(transition_steps, H)

    # Clone forecast so original is not modified
    adjusted = forecast.clone()

    if k > 0:
        # weights: shape (k,)
        w = torch.arange(1, k + 1, device=forecast.device, dtype=forecast.dtype) / k

        # reshape for broadcasting
        w = w.unsqueeze(0)                     # (1, k)
        last_obs_expanded = last_obs.unsqueeze(1)  # (B, 1)

        # Apply blending on first k steps
        adjusted[:, :k] = (
            (1 - w) * last_obs_expanded -
            w * forecast[:, :k]
        )

    return adjusted


# %%

# Plot mean of absolute error with adjusted forecast

y_pred_adj = linear_blend_bridge_batch(y_true[:,-1,:].squeeze(), y_pred.squeeze())
abs_error = abs(y_pred_adj.unsqueeze(dim=2)-y_true)
meanError = abs_error.mean(dim=0)
meanError = meanError.detach().cpu().numpy()

plt.plot(meanError)
plt.title("SM abs Error (mean) adjusted"); plt.xlabel("time forecast Horizon"); plt.ylabel("variance")
plt.grid(True, alpha=0.3)
plt.show()
# %%
