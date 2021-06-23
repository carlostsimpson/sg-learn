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

from prepare_input_layer import PrepareInputLayer


class SGNetProcess(nn.Module):
    def __init__(self, pp):
        super(SGNetProcess, self).__init__()
        #
        self.pp = pp
        self.a = self.pp.alpha
        self.bz = self.pp.betaz
        #
        #
        self.lrl = nn.LeakyReLU()
        #
        self.prep = PrepareInputLayer(self.pp)
        #
        self.channels = self.prep.channels
        #
        self.n = self.pp.model_n
        n = self.n
        #
        self.array_channels = n
        self.process_channels = self.array_channels * self.a * self.a
        #
        self.convA1 = nn.Conv2d(self.channels, 8 * n, 1)
        self.convA2 = nn.Conv2d(
            8 * n,
            8 * n,
            [5, 1],
            padding=[2, 0],
            padding_mode="circular",
            groups=n,
        )
        self.convA3 = nn.Conv2d(
            8 * n,
            8 * n,
            [1, 5],
            padding=[0, 2],
            padding_mode="circular",
            groups=n,
        )
        self.convA4 = nn.Conv2d(
            8 * n,
            8 * n,
            [5, 1],
            padding=[2, 0],
            padding_mode="circular",
            groups=n,
        )
        self.convA5 = nn.Conv2d(
            8 * n,
            8 * n,
            [1, 5],
            padding=[0, 2],
            padding_mode="circular",
            groups=n,
        )
        self.convA5tg = nn.Conv2d(40 * n, 8 * n, 1)
        self.convA6 = nn.Conv2d(
            8 * n,
            8 * n,
            [5, 1],
            padding=[2, 0],
            padding_mode="circular",
            groups=n,
        )
        self.convA7 = nn.Conv2d(
            9 * n,
            8 * n,
            [1, 5],
            padding=[0, 2],
            padding_mode="circular",
            groups=n,
        )
        self.convA8 = nn.Conv2d(
            8 * n,
            8 * n,
            [5, 1],
            padding=[2, 0],
            padding_mode="circular",
            groups=n,
        )
        self.convA9 = nn.Conv2d(
            8 * n,
            8 * n,
            [1, 5],
            padding=[0, 2],
            padding_mode="circular",
            groups=n,
        )
        #
        self.convB = nn.Conv2d(40 * n, n, 1)
        #
        self.convC1 = nn.Conv1d(
            8 * n * self.a * self.a, 8 * n, 1, groups=8 * n
        )
        self.convC2 = nn.Conv1d(8 * n, n * self.a * self.a, 1, groups=n)
        #
        ##

    def forward(self, Data):
        #
        initial_data, _ = self.prep(Data)
        length = Data["length"]
        n = self.n
        #
        yA1 = self.lrl(self.convA1(initial_data))
        yA2 = self.lrl(self.convA2(yA1))
        yA3 = self.lrl(self.convA3(yA2))
        yA4 = self.lrl(self.convA4(yA3))
        yA5 = self.lrl(self.convA5(yA4))
        yA5tg = self.lrl(
            self.convA5tg(torch.cat((yA1, yA2, yA3, yA4, yA5), 1))
        )
        yA6 = self.lrl(self.convA6(yA5tg))
        #
        yA4side = yA4.view(length, 8 * n * self.a * self.a, 1)
        yC1 = self.lrl(self.convC1(yA4side))
        yC2 = self.lrl(self.convC2(yC1)).view(length, n, self.a, self.a)
        yA6side = torch.cat((yA6, yC2), 1)
        #
        yA7 = self.lrl(self.convA7(yA6side))
        yA8 = self.lrl(self.convA8(yA7))
        yA9 = self.lrl(self.convA9(yA8))
        #
        yB = self.lrl(self.convB(torch.cat((yA1, yA3, yA5tg, yA7, yA9), 1)))
        #
        return yB.view(length, self.process_channels)
