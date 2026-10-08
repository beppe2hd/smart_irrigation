from dotenv import load_dotenv
import sys, os

load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))

import torch
from torch import nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )

    def forward(self, x):
        # x: (batch_size, input_seq_len, input_size)
        outputs, hidden = self.lstm(x)  # hidden: (1, batch, hidden_size)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, forecast_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=output_size + forecast_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, decoder_input, hidden):
        # decoder_input: (batch_size, 1, output_size) ← one timestep
        output, hidden = self.lstm(decoder_input, hidden)
        output = self.fc(output)  # (batch_size, 1, output_size)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size, forecast_size, output_seq_len
    ):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(output_size, hidden_size, forecast_size)
        self.output_seq_len = output_seq_len

    def forward(self, x, x_f):
        batch_size = x.size(0)
        output_size = self.decoder.fc.out_features

        x[:, :, 0] *= 5.0
        x[:, :, 1] *= 5.0

        hidden = self.encoder(x)

        # Start with zeros or a special <START> token
        decoder_input = torch.zeros(batch_size, 1, output_size, device=x.device)

        outputs = []

        for i in range(self.output_seq_len):
            current_x_f = x_f[:, i, :].unsqueeze(dim=1)
            decoder_input = torch.cat((decoder_input, current_x_f), dim=2)
            output, hidden = self.decoder(decoder_input, hidden)
            outputs.append(output)
            decoder_input = output  # Teacher forcing could go here

        return torch.cat(outputs, dim=1)
