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

from constants import Dvc
from utils import CoherenceError, arangeic, binaryzbatch, itp, zbinary


class SymmetricGroup:
    def __init__(self, p):
        #
        if p < 1:
            print("can't initialize a symmetric group with size", p)
            raise CoherenceError("exiting")
        if p > 9:
            print(
                "symmetric group size",
                p,
                "is probably going to cause a memory overflow somewhere",
            )
            raise CoherenceError("exiting")
        #
        self.p = p
        # self.subgroups_max = subgroups_max # this is not currently used
        #
        self.gtlength = 1
        for i in range(self.p):
            self.gtlength *= i + 1
        self.grouptable = self.makegrouptable()
        self.gtbinary = self.makegrouptablebinary()
        self.inversetable = self.makeinversetable()

    def symmetricgrouptable(self, k):
        assert k > 0
        if k == 1:
            sgt = torch.zeros((1), dtype=torch.int64, device=Dvc)
            return 1, sgt
        length_prev, sgtprev = self.symmetricgrouptable(k - 1)
        length = length_prev * k
        krange = torch.arange(
            (k), dtype=torch.int64, device=Dvc
        )  # same as arangeic(k)
        krangevx = krange.view(k, 1).expand(k, length_prev)
        krangevx2 = krange.view(k, 1, 1).expand(k, length_prev, k - 1)
        #
        sgtprev_vx = sgtprev.view(1, length_prev, k - 1).expand(
            k, length_prev, k - 1
        )
        #
        krange1vxr = krange.view(k, 1).expand(k, k).reshape(k * k)
        krange2vxr = krange.view(1, k).expand(k, k).reshape(k * k)
        gappedtablev = krange2vxr[(krange1vxr != krange2vxr)]
        gappedtable = gappedtablev.view(k, k - 1)
        #
        afterpart = gappedtable[krangevx2, sgtprev_vx]
        beforepart = krange.view(k, 1, 1).expand(k, length_prev, 1)
        newtablev = torch.cat((beforepart, afterpart), 2)
        newtable = newtablev.view(length, k)
        return length, newtable

    def makegrouptable(self):
        length, table = self.symmetricgrouptable(self.p)
        assert length == self.gtlength
        # print("making group table for symmetric group, as an array of shape",table.size())
        return table

    def makeinversetable(self):
        gl = self.gtlength
        p = self.p
        tablevx = self.grouptable.view(gl, p, 1).expand(gl, p, p)
        yrangevx = arangeic(p).view(1, 1, p).expand(gl, p, p)
        delta = (tablevx == yrangevx).to(torch.int64)
        values, inversetable = torch.max(delta, 1)
        return inversetable


    def makegrouptablebinary(self):
        p = self.p
        gl = self.gtlength
        if p > 7:
            print("warning: not making binary table for p=", itp(p), "> 7")
            return None
        blength = 2 ** p
        brange = arangeic(blength)
        zbinarytable = torch.zeros((blength, p), dtype=torch.bool, device=Dvc)
        for z in range(blength):
            zbinarytable[z, :] = zbinary(p, z)
        gtb = (
            self.grouptable.view(gl, 1, p)
            .expand(gl, blength, p)
            .reshape(gl * blength, p)
        )
        brange = (
            arangeic(blength)
            .view(1, blength, 1)
            .expand(gl, blength, p)
            .reshape(gl * blength, p)
        )
        #
        gtb_mod = zbinarytable[brange, gtb]
        #
        gt_binaryv = binaryzbatch(gl * blength, p, gtb_mod)
        gt_binary = gt_binaryv.view(gl, blength)
        # print("made group table binary")
        return gt_binary
