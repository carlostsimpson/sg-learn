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


class OutputLayer2d(nn.Module):
    def __init__(self, pp, channels_array, channels_mid, channels_extra):
        super(OutputLayer2d, self).__init__()
        self.pp = pp
        self.a = self.pp.alpha
        self.channels_array = channels_array
        self.channels_in = self.channels_array * self.a * self.a
        self.channels_mid = channels_mid
        self.channels_extra = channels_extra
        self.lrl = nn.LeakyReLU()
        self.layerD1 = nn.Linear(self.channels_in, channels_mid, 1)
        self.layerD2 = nn.Linear(channels_mid, channels_mid, 1)
        self.layerD3 = nn.Linear(
            2 * channels_mid, channels_extra * self.a * self.a, 1
        )
        self.layerD4 = nn.Conv2d(
            self.channels_array + self.channels_extra, self.channels_mid, 1
        )
        self.layerD5 = nn.Conv2d(self.channels_mid, self.channels_mid, 1)
        self.layerD6 = nn.Conv2d(
            2 * self.channels_mid + self.channels_extra, 1, 1
        )

    def forward(self, layer_data):
        length = layer_data.size()[0]
        layer_in = layer_data.view(length, self.channels_in)
        yD1 = self.lrl(self.layerD1(layer_in))
        yD2 = self.lrl(self.layerD2(yD1))
        yD3 = self.layerD3(torch.cat((yD1, yD2), 1))
        yD3v = yD3.view(length, self.channels_extra, self.a, self.a)
        layer_array = layer_data.view(
            length, self.channels_array, self.a, self.a
        )
        yDcat = torch.cat((yD3v, layer_array), 1)
        yD4 = self.lrl(self.layerD4(yDcat))
        yD5 = self.lrl(self.layerD5(yD4))
        yD6 = self.layerD6(torch.cat((yD3v, yD4, yD5), 1))
        yScore = yD6.view(length, self.a, self.a)
        return yScore
