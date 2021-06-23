"""
    Machine learning proofs for classification of nilpotent semigroups.
    Copyright (C) 2021  Carlos Simpson

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import torch
from torch import nn

from output_layer_scalar import OutputLayerScalar
from sgnet_process import SGNetProcess


class SGNetGlobal(nn.Module):
    def __init__(self, pp):
        super(SGNetGlobal, self).__init__()
        self.pp = pp
        self.a = self.pp.alpha
        self.process = SGNetProcess(self.pp)
        self.outlayer = OutputLayerScalar(
            self.pp, self.process.process_channels, 32
        )

    def forward(self, Data):
        # see the PrepareInputLayer() class above for the input template assertions on Data
        yProcessed = self.process(Data)
        yScore = self.outlayer(yProcessed)
        #############################
        # output template assertions:
        length = Data["length"]
        assert yScore.dtype == torch.float
        assert yScore.size() == torch.Size([length])
        #############################
        return yScore
