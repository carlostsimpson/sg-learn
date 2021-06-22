import torch
from torch import nn


class OutputLayerScalar(nn.Module):
    def __init__(self, pp, channels_in, channels_mid):
        super(OutputLayerScalar, self).__init__()
        self.pp = pp
        self.a = self.pp.alpha
        #
        self.channels_in = channels_in
        self.channels_mid = channels_mid
        #
        self.lrl = nn.LeakyReLU()
        #
        self.layerD1 = nn.Linear(channels_in, channels_mid, 1)
        self.layerD2 = nn.Linear(channels_mid, channels_mid, 1)
        self.layerD3 = nn.Linear(2 * channels_mid, 1, 1)

    def forward(self, layer_data):
        #
        length = layer_data.size()[0]
        #
        layer_in = layer_data.view(length, self.channels_in)
        yD1 = self.lrl(self.layerD1(layer_in))
        yD2 = self.lrl(self.layerD2(yD1))
        yD3 = self.layerD3(torch.cat((yD1, yD2), 1))
        #
        yScore = yD3.view(length)
        #
        return yScore
