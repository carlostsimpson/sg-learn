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
from utils import arangeic, itt, nump, zbinary


class Relations1:
    def __init__(self, pp):
        #
        self.pp = pp
        #
        self.alpha = self.pp.alpha
        self.alpha2 = self.alpha * self.alpha
        self.alpha3 = self.alpha * self.alpha * self.alpha
        self.alpha3z = self.alpha3 + 1
        self.beta = self.pp.beta
        self.betaz = self.beta + 1
        #
        # self.model = self.mm
        #
        self.qvalue = self.pp.qvalue
        #
        self.ascore_max = self.pp.ascore_max
        #
        self.infosize = self.pp.infosize
        self.pastsize = self.pp.pastsize
        self.futuresize = self.pp.futuresize
        #
        #
        self.ar1 = (
            arangeic(self.alpha)
            .view(self.alpha, 1)
            .expand(self.alpha, self.alpha)
            .clone()
        )
        self.ar2 = (
            arangeic(self.alpha)
            .view(1, self.alpha)
            .expand(self.alpha, self.alpha)
            .clone()
        )
        self.ida = self.ar1 == self.ar2
        #
        self.a3r1 = (
            arangeic(self.alpha3)
            .view(self.alpha3, 1)
            .expand(self.alpha3, self.alpha3)
            .clone()
        )
        self.a3r2 = (
            arangeic(self.alpha3)
            .view(1, self.alpha3)
            .expand(self.alpha3, self.alpha3)
            .clone()
        )
        self.eqa3 = self.a3r1 == self.a3r2
        #
        self.a3zr1 = (
            arangeic(self.alpha3z)
            .view(self.alpha3z, 1)
            .expand(self.alpha3z, self.alpha3z)
            .clone()
        )
        self.a3zr2 = (
            arangeic(self.alpha3z)
            .view(1, self.alpha3z)
            .expand(self.alpha3z, self.alpha3z)
            .clone()
        )
        self.eqa3z = self.a3zr1 == self.a3zr2
        #
        self.iblength = 2 ** (2 * self.beta)
        self.ibarray = torch.zeros(
            (self.iblength, 2 * self.beta), dtype=torch.bool, device=Dvc
        )
        for z in range(self.iblength):
            self.ibarray[z] = zbinary(2 * self.beta, z)
        #
        self.betazsubsets = (
            self.makebetazsubsets()
        )  # at location j,:,: it is for size (j+1)
        self.quantities = (
            self.betazsubsets[:, 0 : self.beta].to(torch.int).sum(1)
        )  # the size of the subset as a function of z
        #

    # general manipulation of data

    def printprod(self, Data, i):
        length = Data["length"]
        prod = Data["prod"]
        a = self.alpha
        bz = self.betaz
        #
        assert i < length
        #
        printarray = torch.zeros((a, a), dtype=torch.int, device=Dvc)
        printarray += 9 * (10 ** bz)
        for p in range(bz):
            printarray += (10 ** p) * (prod[i, :, :, p].to(torch.int))
        print(nump(printarray))

    def makebetazsubsets(self):
        b = self.beta
        bz = self.betaz
        bpower = 2 ** b
        subsets = torch.ones((bpower, bz), dtype=torch.bool, device=Dvc)
        for z in range(bpower):
            subsets[z, 0:b] = zbinary(b, z)
        return subsets

    def nulldata(self):
        length = torch.tensor(0)
        Output = {
            "length": length,
            "depth": None,
            "prod": None,
            "left": None,
            "right": None,
            "ternary": None,
            "info": None,
        }
        return Output

    def copydata(self, Data):
        if Data["length"] == 0:
            return self.nulldata()
        Output = {}
        Output["length"] = itt(Data["length"]).clone().detach()
        for ky in Data.keys():
            if ky != "length":
                Output[ky] = (Data[ky]).clone().detach()
        return Output

    def deletedata(self, Data):
        # del Data['length']  # better avoid doing that
        datakeyslist = list(Data.keys())
        for ky in datakeyslist:
            if ky != "length":
                del Data[ky]
        del Data

    def appenddata(
        self, Data1, Data2
    ):  # appends Data2 to Data1 and outputs the result
        # there is a case where Data1 == None then we just output Data2
        assert set(Data1.keys()) == set(Data2.keys())
        if Data1["length"] == 0:
            return self.copydata(Data2)
        if Data2["length"] == 0:
            return self.copydata(Data1)
        Output = {}
        Output["length"] = Data1["length"] + Data2["length"]
        #
        for ky in Data1.keys():
            if ky != "length":
                Output[ky] = torch.cat((Data1[ky], Data2[ky]), 0)
        return Output

    def indexselectdata(self, Data, indices):
        #
        if len(indices) == 0:
            return self.nulldata()
        #
        Output = {}
        #
        Output["length"] = len(indices)
        #
        for ky in Data.keys():
            if ky != "length":
                Output[ky] = (Data[ky])[indices].clone().detach()
        #
        return Output

    def detectsubdata(self, Data, detection):
        #
        assert len(detection) == Data["length"]
        #
        sublength = detection.to(torch.int).sum(0)
        if sublength == 0:
            return self.nulldata()
        #
        Output = {}
        #
        Output["length"] = sublength
        #
        for ky in Data.keys():
            if ky != "length":
                Output[ky] = (Data[ky])[detection].detach()
        #
        return Output

    def insertdata(self, Data, detection, SubData):
        #
        assert set(Data.keys()) == set(SubData.keys())
        #
        sublength = SubData["length"]
        assert detection.to(torch.int).sum(0) == sublength
        #
        if sublength == 0:
            return Data
        #
        Output = {}
        Output["length"] = itt(Data["length"])
        #
        for ky in Data.keys():
            if ky != "length":
                outputitem = Data[ky].clone().detach()
                outputitem[detection] = SubData[ky]
                Output[ky] = outputitem
        #
        return Output

    def knowledge(self, Data):  # now it increases as we refine
        a = self.alpha
        bz = self.betaz
        length = Data["length"]
        prod = Data["prod"]  # mask of shape a.a.bz with boolean values
        left = Data["left"]
        right = Data["right"]
        ternary = Data["ternary"]
        #
        if length == 0:
            zerokn = torch.zeros((1), dtype=torch.int, device=Dvc)
            return zerokn
        #
        output = torch.zeros((length), dtype=torch.int64, device=Dvc)
        output -= prod.to(torch.int64).view(length, a * a * bz).sum(1)
        output -= left.to(torch.int64).view(length, a * bz * 2).sum(1)
        output -= right.to(torch.int64).view(length, bz * a * 2).sum(1)
        output -= ternary.to(torch.int64).view(length, a * a * a * 2).sum(1)
        return output

    def availablexy(self, length, prod):
        a2 = self.alpha2
        prodsum = prod.to(torch.int64).sum(3)
        possible = ((prodsum > 0).all(2)).all(1)
        possiblexy = possible.view(length, 1).expand(length, a2)
        #
        optionalxy = (prodsum > 1).view(length, a2)
        #
        available_xy = possiblexy & optionalxy
        return available_xy

    def availablexyp(self, length, prod):
        a = self.alpha
        bz = self.betaz
        prodsum = prod.to(torch.int64).sum(3)
        possible = ((prodsum > 0).all(2)).all(1)
        possiblexyp = possible.view(length, 1, 1, 1).expand(length, a, a, bz)
        #
        optionalxyp = (
            (prodsum > 1).view(length, a, a, 1).expand(length, a, a, bz)
        )
        #
        available_xyp = prod & possiblexyp & optionalxyp
        return available_xyp

    ###

    def upsplitting(
        self, Data, ivector, xvector, yvector, pvector
    ):  # setting x.y = p
        a = self.alpha
        bz = self.betaz
        if len(ivector) == 0:
            return self.rr1.nulldata()
        UpData = self.indexselectdata(Data, ivector)
        length = UpData["length"]
        prod = UpData["prod"]
        #
        xrangevx = arangeic(a).view(1, a, 1, 1).expand(length, a, a, bz)
        yrangevx = arangeic(a).view(1, 1, a, 1).expand(length, a, a, bz)
        prangevx = arangeic(bz).view(1, 1, 1, bz).expand(length, a, a, bz)
        #
        xvectorvx = xvector.view(length, 1, 1, 1).expand(length, a, a, bz)
        yvectorvx = yvector.view(length, 1, 1, 1).expand(length, a, a, bz)
        pvectorvx = pvector.view(length, 1, 1, 1).expand(length, a, a, bz)
        #
        newprod = prod & (
            (xrangevx != xvectorvx)
            | (yrangevx != yvectorvx)
            | (prangevx == pvectorvx)
        )
        #
        UpData["prod"] = newprod
        UpData["depth"] += 1
        #
        return UpData
