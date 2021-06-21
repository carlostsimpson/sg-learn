import torch
from torch import nn

from output_layer_scalar import OutputLayerScalar
from sgnet_process import SGNetProcess


class SGNetGlobal(nn.Module):
    def __init__(self, pp):
        super(SGNetGlobal, self).__init__()
        #
        self.pp = pp
        self.a = self.pp.alpha
        #
        self.process = SGNetProcess(self.pp)
        #
        self.outlayer = OutputLayerScalar(
            self.pp, self.process.process_channels, 32
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
        assert yScore.dtype == torch.float
        assert yScore.size() == torch.Size([length])
        #############################
        #
        return yScore
