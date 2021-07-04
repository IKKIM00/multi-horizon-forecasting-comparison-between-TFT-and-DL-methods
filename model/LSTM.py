import torch
import torch.nn as nn



class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_layers):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(hidden_dim * seq_len, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, output_dim)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(32)

        self.act = nn.ReLU()

    def forward(self, inp):
        h0, c0 = self.init_hidden(inp)
        if torch.cuda.is_available():
            h0, c0 = h0.cuda(), c0.cuda()
        output, _ = self.lstm(inp, (h0, c0))
        output = self.flatten(output)
        output = self.act(self.bn1(self.fc1(output)))
        output = self.act(self.bn2(self.fc2(output)))
        output = self.act(self.bn3(self.fc3(output)))
        output = self.act(self.bn4(self.fc4(output)))
        output = self.fc5(output)
        return output

    def init_hidden(self, inp):
        b, _, _ = inp.size()
        h0 = torch.zeros(self.num_layers, b, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, b, self.hidden_dim)
        return [t for t in (h0, c0)]


if __name__=="__main__":
    inp = torch.randn(128, 24, 8)
    model = LSTM(8, 256, 1, 24, 2)
    output = model(inp)
    print(output.shape)
    print(output)
