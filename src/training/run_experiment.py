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


def load_train_test(train_fields, test_fields):
    df_train = []
    df_test = []
    for field_id in train_fields:
        df_train.append(pd.read_csv(f"field_f{field_id}.csv"))
    for field_id in test_fields:
        df_test.append(pd.read_csv(f"field_f{field_id}.csv"))
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


def scale_data(df_train, df_test):

    scaler = StandardScaler()
    df_train_concat = pd.concat(df_train, ignore_index=True)
    scaler.fit(df_train_concat)

    for i in range(len(df_train)):
        df_train_np = scaler.transform(df_train[i])
        df_train[i] = pd.DataFrame(
            df_train_np, columns=df_train[i].columns, index=df_train[i].index
        )
    for i in range(len(df_test)):    
        df_test_np = scaler.transform(df_test[i])
        df_test[i] = pd.DataFrame(
            df_test_np, columns=df_test[i].columns, index=df_test[i].index
        )

    return df_train, df_test, scaler


def set_data_loaders(config, df_train, df_test):

    dataloader_train = []
    dataloader_test = []
    for i in range(len(df_train)):
        dataset_train = TimeSeriesDataset(config, df_train[i])
        dataloader_train.append(DataLoader(dataset_train, batch_size=32, shuffle=True))

    for i in range(len(df_test)):
        dataset_test = TimeSeriesDataset(config, df_test[i])
        dataloader_test.append(DataLoader(dataset_test, batch_size=32, shuffle=True))

    return dataloader_train, dataloader_test


def create_sub_df(dfs, selected_columns):
    reduced_dfs = []
    for current_df in dfs:
        if all(item in current_df.columns for item in selected_columns):
            reduced_dfs.append(current_df[selected_columns])
        else:
            raise Exception(
                f"""Not all items are contained in train columns \n
                Train Columns: {current_df.columns} \n
                Selected_columns: {selected_columns}""",
            )


def data_preparation(config):

    path_train = config["files"]["training"]
    path_test = config["files"]["test"]

    df_train, df_test = load_train_test(path_train, path_test)

    selected_columns = (
        config["features"]["input"]["fiedls"]
        + config["features"]["input"]["meteo_historical"]
        + config["features"]["input"]["meteo_forecast"]
        + config["features"]["output"]
    )
    selected_columns = list(dict.fromkeys(selected_columns))

    print("------------")
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

    df_train = create_sub_df(df_train, selected_columns)
    df_test = create_sub_df(df_test, selected_columns)

    df_train, df_test, scaler = scale_data(df_train, df_test)
    dataloaders_train, dataloaders_test = set_data_loaders(config, df_train, df_test)

    return dataloaders_train, dataloaders_test, scaler, output_scale_index, x_scale_index, x_f_scale_index


def train(config, num_of_epoches, dataloaders_train, dataloaders_test, output_scale_index, scaler):

    model = create_model(config)

    optimizer = load_Optimizer(model, config)
    criterion = load_Loss(config)
    epochs = num_of_epoches

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    mse_overEpoches = np.zeros(epochs)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        count = 0
        for dataloader_train in dataloaders_train:
            for x_batch, x_f_batch, y_batch in dataloader_train:
                count += 1
                x_batch = x_batch.type(torch.float32)
                x_f_batch = x_f_batch.type(torch.float32)
                y_batch = y_batch.type(torch.float32)
                # Forward pass
                outputs = model(x_batch, x_f_batch)

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
                mse_s = 0.0
                count = 0
                for dataloader_test in dataloaders_test:
                    for x_batch, x_f_batch, y_batch in dataloader_test:
                        count += 1
                        x_batch = x_batch.type(torch.float32)
                        x_f_batch = x_f_batch.type(torch.float32)
                        y_batch = y_batch.type(torch.float32)
                        # Forward pass
                        outputs = model(x_batch, x_f_batch)

                        ## apply inverse_scale_data(output, col)
                        outputs = inverse_scale_data(outputs, output_scale_index, scaler)
                        y_batch = inverse_scale_data(y_batch, output_scale_index, scaler)

                        loss = criterion(outputs, y_batch)
                        test_loss += loss.item()
                        mse = torch.mean((y_batch - outputs) ** 2)
                        mse_s += mse

                    mse_overEpoches[epoch] = mse_s / count
                    test_losses[epoch] = test_loss / count

    return model, test_losses, mse_overEpoches


def save(config, model, scaler, output_scale_index, x_scale_index, x_f_scale_index, loss_mse, mse_overEpoches, ):

    print(loss_mse)
    print(mse_overEpoches)
    print(output_scale_index)

    folder_path = "./src/weights/" + config["name"] + config["version"].replace(".", "_")
    os.makedirs(folder_path, exist_ok=True)
    print(f"Folder {folder_path}")

    path_model = folder_path + "/weights.pth"
    path_scaler = folder_path + "/scaler.pkl"
    path_out_scaler_indexs = folder_path + "/scaler_out_indexs.pkl"
    path_x_scaler_indexs = folder_path + "/scaler_x_indexs.pkl"
    path_x_f_scaler_indexs = folder_path + "/scaler_x_f_indexs.pkl"

    path_mse = folder_path + "/mse.json"
    torch.save(model.state_dict(), path_model)
    joblib.dump(scaler, path_scaler)
    joblib.dump(output_scale_index, path_out_scaler_indexs)
    joblib.dump(x_scale_index, path_x_scaler_indexs)
    joblib.dump(x_f_scale_index, path_x_f_scaler_indexs)
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
    #config = get_config_file('src/configurations/config_season2_MSE_W_newfieldsV3.yaml')
    print(config)

    folds = config["folds"]
    for i in range(len(folds)):
        test_fields = folds[i]

        train_fields = [
            item
            for j, fold in enumerate(folds)
            if j != i
            for item in fold
        ]

        print(f"Fold {i + 1}")
        print("Train:", train_fields)
        print("Test: ", test_fields)

        for num_of_epoches in config["hyperparameters"]["epoches"]:

            dataloader_train, dataloader_test, scaler, output_scale_index, x_scale_index, x_f_scale_index = data_preparation(
                config, train_fields, test_fields
            )
            model, loss_mse, mse_overEpoches = train(
                config, num_of_epoches, dataloader_train, dataloader_test, output_scale_index, scaler
            )
            save(config, model, scaler, output_scale_index, x_scale_index, x_f_scale_index, loss_mse, mse_overEpoches)
    # dataloader_train, dataloader_test, scaler, output_scale_index, x_scale_index, x_f_scale_index = data_preparation(
    #     config
    # )
    # model, loss_mse, mse_overEpoches = train(
    #     config, dataloader_train, dataloader_test, output_scale_index, scaler
    # )
    # save(config, model, scaler, output_scale_index, x_scale_index, x_f_scale_index, loss_mse, mse_overEpoches)

