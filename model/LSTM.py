import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, output_dim)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)

        self.act = nn.ReLU()

    def forward(self, inp):
        h0, c0 = self.init_hidden()

        output, _ = self.lstm(inp, (h0, c0))
        output = self.flatten(output[:, -1, :])
        output = self.act(self.bn1(self.fc1(output)))
        output = self.act(self.bn2(self.fc2(output)))
        output = self.fc3(output)
        return output

    def init_hidden(self):
        b, _, _ = inp.size()
        h0 = torch.zeros(self.num_layers, b, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, b, self.hidden_dim)
        return [t for t in (h0, c0)]


if __name__=="__main__":
    inp = torch.randn(128, 24, 8)
    model = LSTM(8, 256, 1, 2)
    output = model(inp)
    print(output.shape)
