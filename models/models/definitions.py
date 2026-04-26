import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout      # active between layers since num_layers=2
        )
        self.bn   = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_last    = h[-1]              # last layer's hidden state
        h_last    = self.bn(h_last)
        h_last    = self.drop(h_last)
        return self.fc(h_last).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.25):
        super().__init__()
        pad         = (kernel - 1) * dilation
        self.conv1  = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2  = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.bn1    = nn.BatchNorm1d(out_ch)
        self.bn2    = nn.BatchNorm1d(out_ch)
        self.drop   = nn.Dropout(dropout)
        self.skip   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu   = nn.ReLU()
        self._pad   = pad

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)[..., :-self._pad]))
        out = self.drop(out)
        out = self.relu(self.bn2(self.conv2(out)[..., :-self._pad]))
        out = self.drop(out)
        return self.relu(out + self.skip(x))


class TCNClassifier(nn.Module):
    def __init__(self, input_dim, channels=32, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(input_dim, channels, dilation=1, dropout=dropout),
            TCNBlock(channels,  channels, dilation=2, dropout=dropout),
            TCNBlock(channels,  channels, dilation=4, dropout=dropout),
        )
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x   = x.transpose(1, 2)        # (batch, features, seq_len)
        out = self.net(x)
        out = out[:, :, -1]            # last timestep
        return self.fc(out).squeeze(-1)
