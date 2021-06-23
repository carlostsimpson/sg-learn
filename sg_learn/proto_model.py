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
from torch import nn, optim

from constants import Dvc
from sgnet_global import SGNetGlobal
from utils import arangeic, itt


class ProtoModel:
    def __init__(self, pp, style):
        self.pp = pp
        self.style = style
        self.a = pp.alpha
        self.a3z = self.a * self.a * self.a + 1
        self.b = pp.beta
        self.bz = self.b + 1
        #
        self.random_order = torch.randperm(self.a * self.a)
        self.spiral, self.spiral_mix = self.makespiral()
        self.rays = self.makerays()
        #
        self.network = SGNetGlobal(self.pp).to(Dvc)
        # there is no network2
        self.network2_trainable = False
        #
        # print(self.network)
        print(
            "set up the proto-model network---network2 is not trainable, this is for the benchmark"
        )
        #
        self.benchmark = True
        #
        self.learning_rate = 0.002
        #
        self.optimizer = optim.SGD(
            self.network.parameters(), lr=self.learning_rate, momentum=0.9
        )
        #
        self.criterionA = nn.L1Loss()
        self.criterionB = nn.MSELoss()
        #
        self.softmax = nn.Softmax(dim=1)

    def makespiral(self):
        spiral_order = torch.zeros(
            (self.a, self.a), dtype=torch.int64, device=Dvc
        )
        spiral_order[0, 0] = 0
        count = 1
        for x in range(1, self.a):
            spiral_order[x, x] = count
            count += 1
            for y in range(x):
                spiral_order[x, y] = count
                count += 1
                spiral_order[y, x] = count
                count += 1
        spiral_orderf = spiral_order.to(torch.float) / itt(self.a * self.a).to(
            torch.float
        )
        thresh = self.pp.spiral_mix_threshold
        over_thresh = (spiral_order > thresh).view(self.a * self.a)
        spiral_mix = spiral_order.clone().view(self.a * self.a)
        spiral_mix[over_thresh] = (
            arangeic(self.a * self.a)[over_thresh] + thresh
        )
        spiral_mixf = spiral_mix.to(torch.float) / itt(self.a * self.a).to(
            torch.float
        )
        return spiral_orderf, spiral_mixf

    def makerays(self):
        a = self.a
        ray_order = torch.zeros(
            (self.a, self.a), dtype=torch.int64, device=Dvc
        )
        count = 0
        for x in range(a):
            ray_order[x, x] = count
            count += 1
        for x in range(a - 1):
            for y in range(x + 1, a):
                ray_order[x, y] = count
                count += 1
                ray_order[y, x] = count
                count += 1
        ray_orderf = ray_order.to(torch.float) / itt(a * a).to(torch.float)
        return ray_orderf

    def virtual_score(self, Data):
        length = Data["length"]
        if self.style == "random":
            output = torch.rand((length, self.a, self.a), device=Dvc)
        if self.style == "random_order":
            output = self.random_order.view(1, self.a, self.a).expand(
                length, self.a, self.a
            )
        if self.style == "spiral":
            output = self.spiral.view(1, self.a, self.a).expand(
                length, self.a, self.a
            )
        if self.style == "spiral_mix":
            output = self.spiral_mix.view(1, self.a, self.a).expand(
                length, self.a, self.a
            )
        if self.style == "rays":
            output = self.rays.view(1, self.a, self.a).expand(
                length, self.a, self.a
            )
        return output

    def network2(self, Data):
        #
        a = self.a
        #
        length = Data["length"]
        prod = Data["prod"]
        #
        prodsum = prod.to(torch.int64).sum(3)
        availablexyv = (prodsum > 1).view(length * a * a)
        #
        vsv = self.virtual_score(Data).reshape(length * a * a)
        vsv[~availablexyv] = 100.0
        return vsv.view(length, a * a)
