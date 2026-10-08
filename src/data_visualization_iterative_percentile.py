import joblib, sys, os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))
print(os.getenv("PYTHONPATH"))
from src.commons.utils import get_config_file

tests = ['config_season2_MSE_W_allFIleds22']

iter = 0
for test in tests:

    iter += 1

    config = get_config_file(f"src/configurations/{test}.yaml")
    folder_path = "src/weights/" + config["name"] + config["version"].replace(".", "_")
    path_test_output = folder_path + "/test_output.pkl"
    path_test_y = folder_path + "/test_y.pkl"
    y_pred = joblib.load(path_test_output)
    y_true = joblib.load(path_test_y)

    print(config['name'])

    # Plot mean of absolute error
    abs_error = abs(y_pred-y_true)
    meanError = abs_error.mean(dim=0)
    meanError = meanError.detach().cpu().numpy()
    #plt.plot(meanError, label=config['name'][25:-1])
    qs = [0, 10, 25, 50, 75, 90, 100]
    err_q = np.percentile(abs_error.detach().cpu().numpy(), qs, axis=0)
    err_q = err_q.squeeze()

    t = np.arange(err_q.shape[1])

    plt.figure(figsize=(10, 5))
    plt.plot(t, err_q[3], label='p50')
    plt.plot(t, meanError, label='abs mean error')
    plt.fill_between(t, err_q[2], err_q[4], alpha=0.3, label='p25-p75')
    plt.fill_between(t, err_q[1], err_q[5], alpha=0.2, label='p10-p90')
    plt.plot(t, err_q[0], '--', alpha=0.5, label='min')
    #plt.plot(t, err_q[6], '--', alpha=0.5, label='max')
    #plt.legend()
    #plt.xlabel("Timestep")
    #plt.ylabel("Absolute error")
    #plt.savefig(f"test{config['name']}.png")



    #if iter ==1:
        #plt.figure()
    holdsm_abs_error = abs(y_true-y_true[:,0,:].unsqueeze(1))
    holdsm_abs_error_mean = holdsm_abs_error.mean(dim=0).cpu()

    err_q = np.percentile(holdsm_abs_error.detach().cpu().numpy(), qs, axis=0)
    err_q = err_q.squeeze()

    #plt.figure(figsize=(10, 5))
    plt.plot(t, err_q[3], label='p50 - hold')
    plt.plot(t, holdsm_abs_error_mean, label='abs mean error  - hold')
    plt.fill_between(t, err_q[2], err_q[4], alpha=0.3, label='p25-p75  - hold')
    plt.fill_between(t, err_q[1], err_q[5], alpha=0.2, label='p10-p90  - hold')
    plt.plot(t, err_q[0], '--', alpha=0.5, label='min  - hold')
        #plt.plot(t, err_q[6], '--', alpha=0.5, label='max')
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("Absolute error")
    plt.savefig(f"test{config['name']}.png")
        

