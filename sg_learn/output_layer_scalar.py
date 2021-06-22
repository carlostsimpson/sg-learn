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
