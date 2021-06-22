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
from sgnet_local import SGNetLocal
from utils import itf, itp


class SgModel:
    def __init__(self, pp):
        self.pp = pp
        #
        self.network = SGNetGlobal(self.pp).to(Dvc)
        #
        self.network2 = SGNetLocal(self.pp).to(Dvc)
        #
        self.average_local_loss = itf(1.0)
        #
        # print(self.network)
        print("set up model network and network2")
        #
        self.benchmark = False
        #
        self.learning_rate = 0.002  # was 0.002, then 0.003, ...
        self.momentum = 0.95
        # self.weight_decay = 0.0001
        #
        # self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum = self.momentum, weight_decay = self.weight_decay )
        # self.optimizer2 = optim.SGD(self.network2.parameters(), lr=self.learning_rate, momentum = self.momentum, weight_decay = self.weight_decay )
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
        )
        self.optimizer2 = optim.SGD(
            self.network2.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
        )
        #
        self.criterionCE = nn.CrossEntropyLoss()
        self.criterionA = nn.L1Loss()
        self.criterionB = nn.MSELoss()
        #
        self.network2_trainable = True
        #
        pp.global_params, pp.local_params = self.modelcount()
        #
        self.softmax = nn.Softmax(dim=1)

    def modelcount(self):
        network_param = sum(
            p.numel() for p in self.network.parameters() if p.requires_grad
        )
        print("network parameters", itp(network_param))
        network2_param = sum(
            p.numel() for p in self.network2.parameters() if p.requires_grad
        )
        print("network2 parameters", itp(network2_param))
        return network_param, network2_param

    def tweak_network(self, N, density, epsilon):
        for p in N.parameters():
            if p.requires_grad:
                tirage_density = torch.rand(p.size(), device=Dvc)
                modif = torch.rand(p.size(), device=Dvc)
                modif *= (tirage_density < density).to(torch.float) * epsilon
                factor = modif + 1.0
                with torch.no_grad():
                    p *= factor
        return

    def save_model(
        self, filename
    ):  # tries to save the two model state dicts and optimizer state dicts
        # I haven't tried these but they should probably mostly work and are included for reference
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "network2_state_dict": self.network2.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "optimizer2_state_dict": self.optimizer2.state_dict(),
            },
            filename,
        )
        print("saved to", filename)
        return

    def load_model(self, filename):  # tries to load from the file
        #
        loadedmodels = torch.load(filename)
        network_state_dict = loadedmodels["network_state_dict"]
        network2_state_dict = loadedmodels["network2_state_dict"]
        optimizer_state_dict = loadedmodels["optimizer_state_dict"]
        optimizer2_state_dict = loadedmodels["optimizer2_state_dict"]
        #
        self.network.load_state_dict(network_state_dict)
        self.network2.load_state_dict(network2_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.optimizer2.load_state_dict(optimizer2_state_dict)
        #
        self.network.train()
        self.network2.train()
        print("loaded from", filename)
        return
