import joblib, sys, os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))
print(os.getenv("PYTHONPATH"))
#from src.commons.utils import get_config_file

exps = ['syntetyc_all_LSTM_L1_E10_LOSSMSE_W0_0_1',
'syntetyc_all_LSTM_L1_E5_LOSSMSE_W0_0_1',
'syntetyc_all_LSTM_L1_E8_LOSSMSE_W0_0_1',
'syntetyc_all_LSTM_L2_E10_LOSSMSE_W0_0_1',
'syntetyc_all_LSTM_L2_E5_LOSSMSE_W0_0_1',
'syntetyc_all_LSTM_L2_E8_LOSSMSE_W0_0_1',
'syntetyc_all_LSTM_L3_E10_LOSSMSE_W0_0_1',
'syntetyc_all_LSTM_L3_E5_LOSSMSE_W0_0_1',
'syntetyc_all_LSTM_L3_E8_LOSSMSE_W0_0_1',
'syntetyc_all_LSTM_L5_E10_LOSSMSE_W0_0_1',
'syntetyc_all_LSTM_L5_E5_LOSSMSE_W0_0_1',
'syntetyc_all_LSTM_L5_E8_LOSSMSE_W0_0_1',
'syntetyc_all_RNN_L1_E10_LOSSMSE_W0_0_1',
'syntetyc_all_RNN_L1_E5_LOSSMSE_W0_0_1',
'syntetyc_all_RNN_L1_E8_LOSSMSE_W0_0_1',
'syntetyc_all_RNN_L2_E10_LOSSMSE_W0_0_1',
'syntetyc_all_RNN_L2_E5_LOSSMSE_W0_0_1',
'syntetyc_all_RNN_L2_E8_LOSSMSE_W0_0_1',
'syntetyc_all_RNN_L3_E10_LOSSMSE_W0_0_1',
'syntetyc_all_RNN_L3_E5_LOSSMSE_W0_0_1',
'syntetyc_all_RNN_L3_E8_LOSSMSE_W0_0_1',
'syntetyc_all_RNN_L5_E10_LOSSMSE_W0_0_1',
'syntetyc_all_RNN_L5_E5_LOSSMSE_W0_0_1',
'syntetyc_all_RNN_L5_E8_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_LSTM_L1_E10_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_LSTM_L1_E5_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_LSTM_L1_E8_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_LSTM_L2_E10_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_LSTM_L2_E5_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_LSTM_L2_E8_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_LSTM_L3_E10_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_LSTM_L3_E5_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_LSTM_L3_E8_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_LSTM_L5_E10_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_LSTM_L5_E5_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_LSTM_L5_E8_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_RNN_L1_E10_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_RNN_L1_E5_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_RNN_L1_E8_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_RNN_L2_E10_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_RNN_L2_E5_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_RNN_L2_E8_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_RNN_L3_E5_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_RNN_L3_E8_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_RNN_L5_E5_LOSSMSE_W0_0_1',
'syntetyc_temp_rain_RNN_L5_E8_LOSSMSE_W0_0_1']



# exps = ['real_syntetyc_all_LSTM_L1_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_LSTM_L1_E5_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_LSTM_L1_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_LSTM_L2_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_LSTM_L2_E5_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_LSTM_L2_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_LSTM_L3_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_LSTM_L3_E5_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_LSTM_L3_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_LSTM_L5_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_LSTM_L5_E5_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_LSTM_L5_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_RNN_L1_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_RNN_L1_E5_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_RNN_L1_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_RNN_L2_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_RNN_L2_E5_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_RNN_L2_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_RNN_L3_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_RNN_L3_E5_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_RNN_L3_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_RNN_L5_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_RNN_L5_E5_LOSSMSE_W0_0_1',
# 'real_syntetyc_all_RNN_L5_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_LSTM_L1_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_LSTM_L1_E5_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_LSTM_L1_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_LSTM_L2_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_LSTM_L2_E5_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_LSTM_L2_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_LSTM_L3_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_LSTM_L3_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_LSTM_L5_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_LSTM_L5_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_RNN_L1_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_RNN_L1_E5_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_RNN_L1_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_RNN_L2_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_RNN_L2_E5_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_RNN_L2_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_RNN_L3_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_RNN_L3_E8_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_RNN_L5_E10_LOSSMSE_W0_0_1',
# 'real_syntetyc_temp_rain_RNN_L5_E8_LOSSMSE_W0_0_1']




iter = 0
res = {}
for exp in exps:

    #iter += 1
    #if 'E5' not in exp and 'L1' not in exp:

        folder_path = "src/weights/" + exp
        path_test_output = folder_path + "/test_output.pkl"
        path_test_y = folder_path + "/test_y.pkl"
        y_pred = joblib.load(path_test_output)
        y_true = joblib.load(path_test_y)

        # Plot mean of absolute error
        abs_error = abs(y_pred-y_true)
        meanError = abs_error.mean(dim=0)
        meanError = meanError.detach().cpu().numpy().mean()
        res[exp.replace('_LOSSMSE_W0_0_1','').replace('real_syntetyc_','')] = meanError

abs_error = abs(y_true-y_true[0])
meanError = abs_error.mean(dim=0)
meanError = meanError.detach().cpu().numpy().mean()
res[f"baseline"] = meanError


res[f"xgboost"] = 2.0334407951


# plt.figure(figsize=(10, 5))
# plt.bar(res.keys(), res.values(), color="skyblue")
# plt.xlabel("Label")
# plt.ylabel("Error")
# plt.title("Error by label")
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig(f"test.png")

fig, ax = plt.subplots(figsize=(10, 5))
pattern = ["#4C78A8", "#F58518", "#54A24B"]  # a, b, c
pattern2 = ["#A84C94", "#1C18F5"]
colors = [pattern[i % 3] for i in range(len(list(res.keys()))-2)]
colors.extend(pattern2)

ax.barh(list(res.keys()), list(res.values()), color=colors)
ax.set_xlabel("Error")
ax.set_ylabel("Label")
ax.set_title("Error by label")
ax.tick_params(axis="y", labelsize=5)
ax.grid(axis="x", linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
ax.invert_yaxis()  # optional
plt.tight_layout()
plt.savefig("test_real.png", dpi=300)

