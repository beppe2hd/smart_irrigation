from dotenv import load_dotenv
import sys, os
import torch
import joblib

load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))

from src.commons.data import inverse_scale_data, specific_scale_data

available_models = ["RNN", "LSTM", "LSTM_SMW"]


def create_model(config):

    model_type = config["architecture"]["type"]

    if model_type in available_models:

        if model_type == "RNN":

            from src.commons.architectures.rnn_encoder_decoder import Seq2Seq

            input_size = len(
                config["features"]["input"]["fiedls"]
                + config["features"]["input"]["meteo_historical"]
            )
            output_size = len(config["features"]["output"])
            hidden_size = config["architecture"]["hidden"]
            forecast_size = len(config["features"]["input"]["meteo_forecast"])

            output_seq_len = config["features"]["output_seq_len"]

            model = Seq2Seq(
                input_size, output_size, hidden_size, forecast_size, output_seq_len
            )

            return model
        
        if model_type == "LSTM":

            from src.commons.architectures.lstm_encoder_decoder import Seq2Seq

            input_size = len(
                config["features"]["input"]["fiedls"]
                + config["features"]["input"]["meteo_historical"]
            )
            output_size = len(config["features"]["output"])
            hidden_size = config["architecture"]["hidden"]
            forecast_size = len(config["features"]["input"]["meteo_forecast"])

            output_seq_len = config["features"]["output_seq_len"]

            model = Seq2Seq(
                input_size, output_size, hidden_size, forecast_size, output_seq_len
            )

            return model


        if model_type == "LSTM_SMW":

            from src.commons.architectures.lstm_encoder_decoder_sm_weighted import Seq2Seq

            input_size = len(
                config["features"]["input"]["fiedls"]
                + config["features"]["input"]["meteo_historical"]
            )
            output_size = len(config["features"]["output"])
            hidden_size = config["architecture"]["hidden"]
            forecast_size = len(config["features"]["input"]["meteo_forecast"])

            output_seq_len = config["features"]["output_seq_len"]

            model = Seq2Seq(
                input_size, output_size, hidden_size, forecast_size, output_seq_len
            )

            return model
            


def inference(model, x, x_f, output_scale_index, x_scale_index, x_f_scale_index, scaler, delta_mode, device):
    with torch.inference_mode():

        print("------")
        print(f"x shape: {x.shape}")
        print(f"x [-1,0:2]: {x[-1,0:2]}")
        x_delta = x[-1,0:2]

        x = specific_scale_data(x, x_scale_index, scaler, device)
        x_f = specific_scale_data(x_f, x_f_scale_index, scaler, device)
        print("------")
        print(f"x shape: {x.shape}")
        print(f"x [0,-1,0:2]: {x[0,-1,0:2]}")

        print("*******")

        x = x.squeeze(0)
        x_f = x_f.squeeze(0)
        x = x.to(torch.float32)
        x_f = x_f.to(torch.float32)

        print(f"x shape before inference {x.shape}")

        y = model(x.unsqueeze(0), x_f.unsqueeze(0))
        print(f'y shape out {y.shape}')
        #if delta_mode ==1:
        #            y=y+x[-1,0:2]
        y = inverse_scale_data(y, output_scale_index, scaler, device)
        #if delta_mode ==1:
        #    y=y+x_delta.unsqueeze(0)
        print("*-------*")
        return y.cpu().squeeze().detach().numpy()
    
def load_weights_and_scale(model, config):

    folder_path = "./src/weights/" + config["name"] + config["version"].replace(".", "_")

    path_model = folder_path + "/weights.pth"
    path_scaler = folder_path + "/scaler.pkl"
    path_scaler_out_indexs = folder_path + "/scaler_out_indexs.pkl"
    path_scaler_x_indexs = folder_path + "/scaler_x_indexs.pkl"
    path_scaler_x_f_indexs = folder_path + "/scaler_x_f_indexs.pkl"

    state_dict = torch.load(path_model, map_location="cpu")
    model.load_state_dict(state_dict)

    scaler = joblib.load(path_scaler)
    output_scale_index = joblib.load(path_scaler_out_indexs)
    x_scale_index = joblib.load(path_scaler_x_indexs)
    x_f_scale_index = joblib.load(path_scaler_x_f_indexs)

    return model, scaler, output_scale_index, x_scale_index, x_f_scale_index
