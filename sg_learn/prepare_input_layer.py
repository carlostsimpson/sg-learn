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


class PrepareInputLayer(nn.Module):
    def __init__(self, pp):
        super(PrepareInputLayer, self).__init__()
        self.pp = pp
        self.a = pp.alpha
        self.b = pp.beta
        self.bz = self.b + 1
        #
        self.n = self.pp.model_n
        #
        a = self.a
        bz = self.bz
        self.channels = 5 * bz + 2 * a
        #

    def forward(self, Data):
        a = self.a
        bz = self.bz
        length = Data["length"]
        prod = Data["prod"]
        left = Data["left"]
        right = Data["right"]
        ternary = Data["ternary"]
        #
        ############################
        # input template assertions:
        #
        assert prod.dtype == torch.bool
        assert left.dtype == torch.bool
        assert right.dtype == torch.bool
        assert ternary.dtype == torch.bool
        #
        assert prod.size() == torch.Size([length, a, a, bz])
        assert left.size() == torch.Size([length, a, bz, 2])
        assert right.size() == torch.Size([length, bz, a, 2])
        assert ternary.size() == torch.Size([length, a, a, a, 2])
        #
        ############################
        #
        prod_data = prod.permute(0, 3, 1, 2).view(length, bz, a, a)
        left_data = (
            left.permute(0, 3, 2, 1)
            .reshape(length, 2, bz, a, 1)
            .expand(length, 2, bz, a, a)
        )
        right_data = (
            right.permute(0, 3, 1, 2)
            .reshape(length, 2, bz, 1, a)
            .expand(length, 2, bz, a, a)
        )
        ternary_data = ternary.permute(0, 4, 2, 1, 3).reshape(
            length, 2, a, a, a
        )
        with torch.no_grad():
            prod_f = prod_data.float()
            prod_denom = (
                prod_f.sum(1).view(length, 1, a, a).expand(length, bz, a, a)
            )
            prod_denom = torch.clamp(prod_denom, 1.0, 100.0)
            prod_ren = prod_f / prod_denom
            prod_ren = (bz * prod_ren) - 1.0
            #
            left_f = left_data.float()
            left_denom = (
                left_f.sum(1)
                .view(length, 1, bz, a, a)
                .expand(length, 2, bz, a, a)
            )
            left_denom = torch.clamp(left_denom, 1.0, 100.0)
            left_ren = (left_f / left_denom).view(length, 2 * bz, a, a)
            left_ren = left_ren - 0.5
            #
            right_f = right_data.float()
            right_denom = (
                right_f.sum(1)
                .view(length, 1, bz, a, a)
                .expand(length, 2, bz, a, a)
            )
            right_denom = torch.clamp(right_denom, 1.0, 100.0)
            right_ren = (right_f / right_denom).view(length, 2 * bz, a, a)
            right_ren = right_ren - 0.5
            #
            ternary_f = ternary_data.float()
            ternary_denom = (
                ternary_f.sum(1)
                .view(length, 1, a, a, a)
                .expand(length, 2, a, a, a)
            )
            ternary_denom = torch.clamp(ternary_denom, 1.0, 100.0)
            ternary_ren = (ternary_f / ternary_denom).reshape(
                length, 2 * a, a, a
            )
            ternary_ren = ternary_ren - 0.5
            #
            initial_data = torch.cat(
                (prod_ren, left_ren, right_ren, ternary_ren), 1
            ).float()
        #
        assert initial_data.size() == torch.Size([length, self.channels, a, a])
        #
        return initial_data, prod_data
