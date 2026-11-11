import argparse
import os

from sympy import im
from src.commons.utils import get_config_file
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AI pipeline with configuration file"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (YAML/JSON)"
    )

    return parser.parse_args()

args = parse_args()
config_path = args.config
print(f"running with configuration file: {config_path}")
config = get_config_file(config_path)
print(config)

for num_of_epoches in config["hyperparameters"]["epoches"]:
    for architecture in config["architecture"]["type"]:
        for hidden_layers in config["architecture"]["hidden"]:

            folds = config["folds"]
            mse_results = []
            for current_fold in range(len(folds)):
                folder_path = f"experiments/outputs/{config['name']}_architecture_{architecture}_hidden_{hidden_layers}_epoches_{num_of_epoches}_fold_{current_fold + 1}/mse.json"
                #print(f"Folder {folder_path}")
                with open(folder_path, "r", encoding="utf-8") as f:
                    mse_list = json.load(f)
                mse_results.append(mse_list[-1])

            print(f"Architecture: {architecture}, Hidden Layers: {hidden_layers}, Epoches: {num_of_epoches}")
            print(np.mean(mse_results))
            #print(np.var(mse_results))
            #print(mse_results)