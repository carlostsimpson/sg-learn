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
from historical import Historical
from relations_2 import Relations2
from utils import arangeic, itp, nump


class Relations3:
    def __init__(self, pp, HST: Historical):
        #
        self.pp = pp
        #
        self.rr2 = Relations2(pp)
        self.rr1 = self.rr2.rr1
        #
        self.alpha = self.pp.alpha
        self.alpha2 = self.alpha * self.alpha
        self.alpha2z = self.alpha2 + 1
        self.alpha3 = self.alpha * self.alpha * self.alpha
        self.alpha3z = self.alpha3 + 1
        self.beta = self.pp.beta
        self.betaz = self.beta + 1
        self.HST = HST
        #

    def printmultiplicities(self, Data):
        #
        dlength = self.alpha2 + 1
        #
        length = Data["length"]
        depth = Data["depth"]
        if length == 0:
            multiplicities = torch.zeros(
                (dlength), dtype=torch.int, device=Dvc
            )
            print(nump(multiplicities))
            return
        #
        dmax, dindices = torch.max(depth, 0)
        if dmax + 2 > dlength:
            dlength = dmax + 2
        if dlength > 50:
            dlength = 50
        #
        depthvx = depth.view(length, 1).expand(length, dlength)
        drangevx = arangeic(dlength).view(1, dlength).expand(length, dlength)
        rectangle = depthvx == drangevx
        multiplicities = rectangle.sum(0)
        #
        # print("dlength",itp(dlength))
        print("multiplicities in active pool by depth:")
        print(nump(multiplicities))
        return

    def selectchunk(self, Data):
        #
        length = Data["length"]
        depth = Data["depth"]
        prod = Data["prod"]
        #
        #
        assert length > 0
        #
        prodstats = prod.to(torch.int).sum(3)
        assert (((prodstats > 0).all(2)).all(1)).all(0)
        optional = prodstats > 1
        assert ((optional.any(2)).any(1)).all(0)
        #
        values, indices = torch.sort(depth, 0, descending=True)
        upper = length
        if upper > self.pp.chunksize:
            upper = self.pp.chunksize
        indices_upper = indices[0:upper]
        #
        cdetection = torch.zeros((length), dtype=torch.bool, device=Dvc)
        cdetection[indices_upper] = True
        #
        ChunkData = self.rr1.detectsubdata(Data, cdetection)
        #
        if self.pp.verbose:
            self.printmultiplicities(Data)
        #
        return ChunkData, cdetection

    def network_vcuts(self, M, Data, randomize):
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        length = Data["length"]
        depth = Data["depth"]
        prod = Data["prod"]
        #
        assert length > 0
        #
        #
        prodstats = prod.to(torch.int64).sum(3)
        #
        assert (((prodstats > 0).all(2)).all(1)).all(0)
        #
        #
        availablexyr = self.rr1.availablexy(length, prod).reshape(length * a2)
        #
        networkscorer = M.network2(Data).detach().reshape(length * a2)
        #
        if randomize:
            #
            epsilon_tirage = torch.rand(length * a2, device=Dvc)
            alll = torch.clamp(M.average_local_loss, 0.0, 0.5)
            epsilon_factor = (
                M.average_local_loss
                * self.pp.perturbation_factor
                * (epsilon_tirage ** 4)
            )
            epsilon = torch.rand(length * a2, device=Dvc)
            networkscorer += epsilon_factor * epsilon
            #
            phase = Data["info"][:, self.pp.phase]
            tirage = torch.rand(length, device=Dvc)
            detection = (phase == 1) | ((tirage < 0.1) & (phase == 2))
            detectionvxr = (
                detection.view(length, 1)
                .expand(length, a2)
                .reshape(length * a2)
            )
            randomscore = torch.rand(length * a2, device=Dvc)
            networkscorer[detectionvxr] = randomscore[detectionvxr]
            #
        #
        networkscorer = torch.clamp(networkscorer, -1.0, 10.0)
        networkscorer[~availablexyr] = 20.0
        networkscore = networkscorer.view(length, a2)
        #
        values, xyvector = torch.min(networkscore, 1)
        #
        return xyvector

    def addvalencies(
        self, availablexyp, xyvector
    ):  # adds into the HST file the valencies of these vertices
        # also adds the passive count (just the length of the xyvector)
        #
        a = self.alpha
        a2 = self.alpha2
        a2z = self.alpha2 + 1
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        length = len(xyvector)
        self.HST.current_proof_passive_count += length
        #
        availablexypv = availablexyp.view(length, a2, bz)
        lrange = arangeic(length)
        available_cuts = availablexypv[lrange, xyvector]
        valency = available_cuts.to(torch.int64).sum(1)
        for v in range(bz + 1):
            self.HST.current_proof_valency_frequency[v] += (
                (valency == v).to(torch.int64).sum(0)
            )
        return

    def managesplit(self, M, DataToSplit, randomize):
        #
        a = self.alpha
        a2 = self.alpha2
        a2z = self.alpha2 + 1
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        #
        length = DataToSplit["length"]
        prod = DataToSplit["prod"]
        #
        availablexyp = self.rr1.availablexyp(length, prod).view(length, a2, bz)
        #
        xyvector = self.network_vcuts(M, DataToSplit, randomize)
        #
        self.addvalencies(availablexyp, xyvector)
        #
        lrangevxr = (
            arangeic(length)
            .view(length, 1)
            .expand(length, bz)
            .reshape(length * bz)
        )
        xyvectorvxr = (
            xyvector.view(length, 1).expand(length, bz).reshape(length * bz)
        )
        bzrangevxr = (
            arangeic(bz).view(1, bz).expand(length, bz).reshape(length * bz)
        )
        #
        verticaldetect = availablexyp[lrangevxr, xyvectorvxr, bzrangevxr]
        #
        #
        ivector_vert = lrangevxr[verticaldetect]
        xyvector_vert = xyvectorvxr[verticaldetect]
        pvector_vert = bzrangevxr[verticaldetect]
        #
        prx = arangeic(a).view(a, 1).expand(a, a).reshape(a2)
        pry = arangeic(a).view(1, a).expand(a, a).reshape(a2)
        #
        xvector_vert = prx[xyvector_vert]
        yvector_vert = pry[xyvector_vert]
        #
        NewData = self.rr1.upsplitting(
            DataToSplit, ivector_vert, xvector_vert, yvector_vert, pvector_vert
        )
        #
        #
        ndlength = NewData["length"]
        #
        #
        AssocNewData = self.rr1.nulldata()
        detection = torch.zeros((ndlength), dtype=torch.bool, device=Dvc)
        newactive = torch.zeros((ndlength), dtype=torch.bool, device=Dvc)
        newdone = torch.zeros((ndlength), dtype=torch.bool, device=Dvc)
        newimpossible = torch.zeros((ndlength), dtype=torch.bool, device=Dvc)
        lower = 0
        for i in range(ndlength):
            assert lower < ndlength
            upper = lower + 1000
            if upper > ndlength:
                upper = ndlength
            detection[:] = False
            detection[lower:upper] = True
            NewDataSlice = self.rr1.detectsubdata(NewData, detection)
            AssocNewDataSlice = self.rr2.process(NewDataSlice)
            AssocNewData = self.rr1.appenddata(AssocNewData, AssocNewDataSlice)
            newactive_s, newdone_s, newimpossible_s = self.rr2.filterdata(
                AssocNewDataSlice
            )
            newactive[lower:upper] = newactive_s
            newdone[lower:upper] = newdone_s
            newimpossible[lower:upper] = newimpossible_s
            lower = upper
            if lower >= ndlength:
                break
        #
        NewActiveData = self.rr1.detectsubdata(AssocNewData, newactive)
        #
        NewDoneData = self.rr1.detectsubdata(AssocNewData, newdone)
        #
        if NewActiveData["length"] > 0:
            phase1 = NewActiveData["info"][:, self.pp.phase] == 1
            phase2 = NewActiveData["info"][:, self.pp.phase] == 2
            tirage = torch.rand(NewActiveData["length"], device=Dvc)
            phasechange = phase2 & (tirage < self.pp.splitting_probability)
            newphase = NewActiveData["info"][:, self.pp.phase].clone()
            newphase[phase1] = 0
            newphase[phasechange] = 1
            NewActiveData["info"][:, self.pp.phase] = newphase
        #
        self.HST.current_proof_impossible_count += newimpossible.to(
            torch.int64
        ).sum(0)
        self.HST.current_proof_done_count += newdone.to(torch.int64).sum(0)
        #
        if self.pp.verbose:
            print(" >>>")
            print("DataToSplit", itp(DataToSplit["length"]))
            print("NewData", itp(NewData["length"]))
            print("NewActiveData", itp(NewActiveData["length"]))
            print("NewDoneData", itp(NewDoneData["length"]))
            print("----------------------------------")
        #
        return NewActiveData, NewDoneData
