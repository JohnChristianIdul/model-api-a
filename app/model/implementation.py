import torch
import torch.nn as nn


class TCNSingleBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, dilation, padding, dropout):
        super(TCNSingleBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size,
                              stride=stride, dilation=dilation, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size,
                              stride=stride, dilation=dilation, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        # Match sequence lengths
        if out.size(2) != res.size(2):
            diff = abs(out.size(2) - res.size(2))
            if out.size(2) > res.size(2):
                out = out[:, :, :-diff]
            else:
                res = res[:, :, :-diff]

        return self.relu1(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            self.layers.append(
                TCNSingleBlock(in_channels, out_channels, kernel_size,
                              stride=1, dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size,
                              dropout=dropout)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TCNForecaster(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.1):
        super(TCNForecaster, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: [batch, sequence_length, channels]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # Permute to [batch, channels, sequence_length]
        x = x.permute(0, 1, 2)
        tcn_out = self.tcn(x)
        tcn_out = tcn_out[:, :, -1]
        output = self.linear(tcn_out)

        return output