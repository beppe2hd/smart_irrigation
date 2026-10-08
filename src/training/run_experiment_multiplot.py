# %%
import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from tqdm import tqdm
import joblib
import yaml
import argparse
import os
import sys
import json

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))

from src.commons.trainingObjects.loadObj import load_Optimizer, load_Loss
from src.commons.architectures.model_handler import create_model
from src.commons.utils import get_config_file
from src.commons.data import inverse_scale_data


def set_randomness():
    # Define the seed value
    seed = 42

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # Set seed for CUDA (if using GPUs)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Ensure deterministic behavior for PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_train_test(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


class TimeSeriesDataset(Dataset):
    def __init__(self, config, data):
        """
        Args:
            data (pandas.Dataframe)
        """
        # if isinstance(data, np.ndarray):
        #    data = torch.tensor(data, dtype=torch.float32)

        self.data = data
        self.input_seq_len = config["features"]["input_seq_len"]
        self.output_seq_len = config["features"]["output_seq_len"]
        self.input_features_list = (
            config["features"]["input"]["fiedls"]
            + config["features"]["input"]["meteo_historical"]
        )
        self.input_forecast_features_list = config["features"]["input"][
            "meteo_forecast"
        ]
        self.out_features_list = config["features"]["output"]
        self.shift = config["features"]["shift"]
        self.length = (
            len(self.data) - self.input_seq_len - self.output_seq_len + 1 - self.shift
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[self.input_features_list]
        x = torch.tensor(x.values, dtype=torch.float32)
        x = x[idx : idx + self.input_seq_len, :]

        x_f = self.data[self.input_forecast_features_list]
        x_f = torch.tensor(x_f.values, dtype=torch.float32)
        x_f = x_f[
            idx
            + self.input_seq_len
            + self.shift : idx
            + self.input_seq_len
            + self.output_seq_len
            + self.shift,
            :,
        ]

        y = self.data[self.out_features_list]
        y = torch.tensor(y.values, dtype=torch.float32)
        y = y[
            idx
            + self.input_seq_len
            + self.shift : idx
            + self.input_seq_len
            + self.output_seq_len
            + self.shift,
            :,
        ]

        return x, x_f, y
    
class TimeSeriesDataset_Delta(Dataset):
    def __init__(self, config, data):
        """
        Args:
            data (pandas.Dataframe)
        """
        # if isinstance(data, np.ndarray):
        #    data = torch.tensor(data, dtype=torch.float32)

        self.data = data
        self.input_seq_len = config["features"]["input_seq_len"]
        self.output_seq_len = config["features"]["output_seq_len"]
        self.input_features_list = (
            config["features"]["input"]["fiedls"]
            + config["features"]["input"]["meteo_historical"]
        )
        self.input_forecast_features_list = config["features"]["input"][
            "meteo_forecast"
        ]
        self.out_features_list = config["features"]["output"]
        self.shift = config["features"]["shift"]
        self.length = (
            len(self.data) - self.input_seq_len - self.output_seq_len + 1 - self.shift
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[self.input_features_list]
        x = torch.tensor(x.values, dtype=torch.float32)
        x = x[idx : idx + self.input_seq_len, :]

        x_f = self.data[self.input_forecast_features_list]
        x_f = torch.tensor(x_f.values, dtype=torch.float32)
        x_f = x_f[
            idx
            + self.input_seq_len
            + self.shift : idx
            + self.input_seq_len
            + self.output_seq_len
            + self.shift,
            :,
        ]

        y = self.data[self.out_features_list]
        y = torch.tensor(y.values, dtype=torch.float32)
        y = y[
            idx
            + self.input_seq_len
            + self.shift : idx
            + self.input_seq_len
            + self.output_seq_len
            + self.shift,
            :,
        ]

        y=y-x[-1,0:1]

        return x, x_f, y


def scale_data(df):

    df_np = scaler.transform(df)

    df = pd.DataFrame(df_np, columns=df.columns, index=df.index)

    return df


def set_data_loaders(config, df):

    if config['delta_mode']==0:
        dataset = TimeSeriesDataset(config, df)
    if config['delta_mode']==1:
        dataset = TimeSeriesDataset_Delta(config, df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader


def create_sub_df(df, selected_columns):
    if all(item in df.columns for item in selected_columns):
        return df[selected_columns]
    else:
        raise Exception(
            f"""Not all items are contained in train columns \n
            Train Columns: {df.columns} \n
            Selected_columns: {selected_columns}""",
        )


def data_preparation(config):

    paths_train = config["files"]["training"]
    print(paths_train)

    merged_df = pd.concat([pd.read_csv(file) for file in paths_train], ignore_index=True)

    selected_columns = (
        config["features"]["input"]["fiedls"]
        + config["features"]["input"]["meteo_historical"]
        + config["features"]["input"]["meteo_forecast"]
        + config["features"]["output"]
    )
    selected_columns = list(dict.fromkeys(selected_columns))

    targets = config["features"]["output"]
    output_scale_index = [selected_columns.index(t) for t in targets]
    targets_fields = config["features"]["input"]["fiedls"]
    target_mh = config["features"]["input"]["meteo_historical"]
    targets = []
    targets.extend(targets_fields)
    targets.extend(target_mh)
    x_scale_index = [selected_columns.index(t) for t in targets]
    targets = config["features"]["input"]["meteo_forecast"]
    x_f_scale_index = [selected_columns.index(t) for t in targets]

    merged_df = create_sub_df(merged_df, selected_columns)
    scaler = StandardScaler()
    scaler.fit(merged_df)

    return scaler, output_scale_index, x_scale_index, x_f_scale_index, selected_columns


def train(config,output_scale_index, scaler, selected_columns):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    model = create_model(config)
    model.to(device)

    optimizer = load_Optimizer(model, config)
    criterion = load_Loss(config, device)
    epochs = config["hyperparameters"]["epoches"]

    sectors = [0,6,12,24,48]
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    mse_overEpoches = np.zeros([epochs,len(sectors)-1])
    tensor_list_y = []
    tensor_list_output = []


    df_test = pd.read_csv(config["files"]["test"])
    df_test = create_sub_df(df_test, selected_columns)
    df_test = scale_data(df_test)
    dataloader_test = set_data_loaders(config, df_test)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        count = 0
        used_plots = 0
        
        for plot_file_path in config["files"]["training"]:

            used_plots+=1
            print(f'Currently working on plot {plot_file_path.split("/")[-1]} - plot {used_plots} of {len(config["files"]["training"])}')

            df_train = pd.read_csv(plot_file_path)
            df_train = create_sub_df(df_train, selected_columns)
            df_train = scale_data(df_train)
            dataloader_train = set_data_loaders(config, df_train)

            for x_batch, x_f_batch, y_batch in dataloader_train:
                count += 1
                x_batch = x_batch.type(torch.float32).to(device)
                x_f_batch = x_f_batch.type(torch.float32).to(device)
                y_batch = y_batch.type(torch.float32).to(device)
                #print(x_batch)
                #print(x_f_batch)
                #print(y_batch)
                # Forward pass
                outputs = model(x_batch, x_f_batch)
                #print(outputs)

                loss = criterion(outputs, y_batch)
                
                train_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        train_losses[epoch] = train_loss / count

        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_loss = 0.0
            mse_s = np.zeros(len(sectors)-1)
            count = 0
            for x_batch, x_f_batch, y_batch in dataloader_test:
                count += 1
                x_batch = x_batch.type(torch.float32).to(device)
                x_f_batch = x_f_batch.type(torch.float32).to(device)
                y_batch = y_batch.type(torch.float32).to(device)
                # Forward pass
                outputs = model(x_batch, x_f_batch)

                ## apply inverse_scale_data(output, col)
                outputs = inverse_scale_data(outputs, output_scale_index, scaler, device)
                y_batch = inverse_scale_data(y_batch, output_scale_index, scaler, device)
                if epoch == epochs-1:
                    tensor_list_y.append(y_batch)
                    tensor_list_output.append(outputs)

                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
                for i in range(len(sectors)-1):
                    mse_s[i] += torch.mean((y_batch[:,sectors[i]:sectors[i+1],:] - outputs[:,sectors[i]:sectors[i+1],:]) ** 2)
            if epoch == epochs-1:
                test_y = torch.cat(tensor_list_y, dim=0)
                test_output = torch.cat(tensor_list_output, dim=0)
        mse_overEpoches[epoch,:] = mse_s / count
        test_losses[epoch] = test_loss / count

    return model, test_losses, mse_overEpoches, test_y, test_output


def save(config, model, scaler, output_scale_index, x_scale_index, x_f_scale_index, loss_mse, mse_overEpoches, test_y, test_output):

    print(loss_mse)
    print(mse_overEpoches)
    print(output_scale_index)

    name = f'{config["name"]}_{config["architecture"]["type"]}_L{config["architecture"]["hidden"]}_E{config["hyperparameters"]["epoches"]}_LOSS{config["hyperparameters"]["loss"]}'

    folder_path = "./src/weights/" + name + config["version"].replace(".", "_")
    os.makedirs(folder_path, exist_ok=True)
    print(f"Folder {folder_path}")

    path_model = folder_path + "/weights.pth"
    path_scaler = folder_path + "/scaler.pkl"
    path_out_scaler_indexs = folder_path + "/scaler_out_indexs.pkl"
    path_x_scaler_indexs = folder_path + "/scaler_x_indexs.pkl"
    path_x_f_scaler_indexs = folder_path + "/scaler_x_f_indexs.pkl"
    path_test_y = folder_path + "/test_y.pkl"
    path_test_output = folder_path + "/test_output.pkl"

    path_mse = folder_path + "/mse.json"
    torch.save(model.state_dict(), path_model)
    joblib.dump(scaler, path_scaler)
    joblib.dump(output_scale_index, path_out_scaler_indexs)
    joblib.dump(x_scale_index, path_x_scaler_indexs)
    joblib.dump(x_f_scale_index, path_x_f_scaler_indexs)
    joblib.dump(test_y, path_test_y)
    joblib.dump(test_output, path_test_output)

    with open(path_mse, "w") as f:
        json.dump(mse_overEpoches.tolist(), f)
    ## complete with a text file reporting information on loss

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


if __name__ == "__main__":
    set_randomness()
    args = parse_args()
    config_path = args.config
    print(f"running with configuration file: {config_path}")
    config = get_config_file(config_path)
    #config = get_config_file('src/configurations/config_season2_MSE_W_allFIleds0.yaml')
    print(config)

    for archType in ["RNN", "LSTM"]:
        for layers in [1, 3, 5]:
            for epo in [5,10]:
                config["architecture"]["type"]=archType
                config["hyperparameters"]["epoches"]=epo
                config["architecture"]["hidden"]=layers

            
                scaler, output_scale_index, x_scale_index, x_f_scale_index, selected_columns = data_preparation(
                    config
                )
                model, loss_mse, mse_overEpoches, test_y, test_output = train(
                    config, output_scale_index, scaler, selected_columns
                )
                save(config, model, scaler, output_scale_index, x_scale_index, x_f_scale_index, loss_mse, mse_overEpoches, test_y, test_output)

