import torch
from torch import nn

from output_layer_2d import OutputLayer2d
from sgnet_process import SGNetProcess


class SGNetLocal(nn.Module):
    def __init__(self, pp):
        super(SGNetLocal, self).__init__()
        # an affine operation: y = Wx + b
        #
        self.pp = pp
        self.a = self.pp.alpha
        #
        self.process = SGNetProcess(self.pp)
        #
        self.outlayer = OutputLayer2d(
            self.pp, self.process.array_channels, 32, 4
        )

    def forward(self, Data):
        #
        # see the PrepareInputLayer() class above for the input template assertions on Data
        #
        yProcessed = self.process(Data)
        #
        yScore = self.outlayer(yProcessed)
        #
        #############################
        # output template assertions:
        length = Data["length"]
        a = self.a
        assert yScore.dtype == torch.float
        assert yScore.size() == torch.Size([length, a, a])
        #
        #############################
        #
        return yScore
