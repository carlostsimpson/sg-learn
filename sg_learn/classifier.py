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
from relations_4 import Relations4
from symmetric_group import SymmetricGroup
from utils import arangeic, itf, itp, numpr


class Classifier:  # this is a very first part of classification up to isomorphism
    def __init__(self, P, HST: Historical):
        #
        #
        self.Pp = P
        self.rr4 = Relations4(self.Pp, HST)
        self.rr3 = self.rr4.rr2
        self.rr2 = self.rr4.rr2
        self.rr1 = self.rr4.rr1
        self.alpha = self.Pp.alpha
        self.alpha2 = self.Pp.alpha2
        self.alpha3 = self.Pp.alpha3
        self.alpha3z = self.Pp.alpha3z
        self.beta = self.Pp.beta
        self.betaz = self.Pp.betaz
        #
        self.sga = SymmetricGroup(self.alpha)
        #
        self.eqlength = 0
        self.eqlist = None
        #
        # self.iota = 24*24*4
        self.matrix = self.choosematrix()
        self.ilength = 10
        (
            self.indices1,
            self.indices2,
            self.indices3,
            self.indices4,
            self.vector,
        ) = self.choosestuff(self.ilength)

    def initialize(self):
        self.eqlength = 0
        self.eqlist = None
        return

    def choosestuff(self, ilength):
        assert ilength <= 48
        permutation1 = torch.randperm(48, device=Dvc)
        permutation2 = torch.randperm(48, device=Dvc)
        permutation3 = torch.randperm(48, device=Dvc)
        permutation4 = torch.randperm(48, device=Dvc)
        #
        indices1 = permutation1[0:ilength]
        indices2 = permutation2[0:ilength]
        indices3 = permutation3[0:ilength]
        indices4 = permutation4[0:ilength]
        #
        permutationV = torch.randperm(101, device=Dvc)
        vector = permutationV[0:ilength] - 50
        return indices1, indices2, indices3, indices4, vector

    def choosematrix(self):
        #
        a = self.alpha
        #
        # thematrix = torch.zeros((a,a,a,a),dtype = torch.int64,device=Dvc)
        for i in range(10):
            permutation = torch.randperm((a * a * a * a), device=Dvc)
            psquared = permutation * permutation
            thematrix = psquared.view(a, a, a, a)
        return thematrix

    def geteq(self, Data):
        #
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a4 = self.alpha2 * self.alpha2
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        length = Data["length"]
        prod = Data["prod"]
        #
        prodsum = prod.to(torch.int64).sum(3)
        #
        assert (((prodsum == 1).all(2)).all(1)).all(0)
        #
        values, table = torch.max(prod.to(torch.int64), 3)
        #
        table1vx = table.view(length, a, a, 1, 1).expand(length, a, a, a, a)
        table2vx = table.view(length, 1, 1, a, a).expand(length, a, a, a, a)
        eq = table1vx == table2vx
        iz = (table == b).view(length, a, a)
        return eq, iz

    def orderinvariantSlice(self, Data):
        #
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a4 = self.alpha2 * self.alpha2
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        length = Data["length"]
        # print("at start of orderinvariantSlice, length is",length)
        prod = Data["prod"]
        #
        eq, iz = self.geteq(Data)
        #
        eqview = eq.view(1, length, a, a, a, a)
        izview = iz.view(1, length, a, a, 1, 1).expand(1, length, a, a, a, a)
        eqiz = torch.cat((eqview, izview), 0)
        eqiz_p2 = eqiz.permute(0, 1, 2, 3, 5, 4)
        eqiz_p3 = eqiz.permute(0, 1, 2, 4, 3, 5)
        eqiz_p4 = eqiz.permute(0, 1, 2, 4, 5, 3)
        eqiz_p5 = eqiz.permute(0, 1, 2, 5, 3, 4)
        eqiz_p6 = eqiz.permute(0, 1, 2, 5, 4, 3)
        #
        eqiz_c1 = torch.cat(
            (eqiz, eqiz_p2, eqiz_p3, eqiz_p4, eqiz_p5, eqiz_p6), 0
        )
        eqiz_c2 = eqiz_c1.permute(0, 1, 3, 4, 5, 2)
        eqiz_c3 = eqiz_c1.permute(0, 1, 4, 5, 2, 3)
        eqiz_c4 = eqiz_c1.permute(0, 1, 5, 2, 3, 4)
        #
        eqiz_cat = torch.cat(
            (eqiz_c1, eqiz_c2, eqiz_c3, eqiz_c4), 0
        )  # size 48.length.a.a.a.a
        #
        eqiz_cat1 = eqiz_cat[self.indices1]
        eqiz_cat2 = eqiz_cat[self.indices2]
        eqiz_cat3 = eqiz_cat[self.indices3]
        eqiz_cat4 = eqiz_cat[self.indices4]
        #
        eqiz_andor = (eqiz_cat1 | eqiz_cat2) & (eqiz_cat3 | eqiz_cat4)
        #
        eqiz_sum = ((eqiz_andor.to(torch.int64).sum(5)).sum(4)).sum(3)
        #
        vectorvx = self.vector.view(self.ilength, 1, 1).expand(
            self.ilength, length, a
        )
        invariant = (eqiz_sum * vectorvx).sum(0)
        #
        return invariant

    def orderinvariant(self, Data):
        #
        a = self.alpha
        #
        length = Data["length"]
        #
        lrange = arangeic(length)
        #
        invariant = torch.zeros((length, a), dtype=torch.int64, device=Dvc)
        #
        lower = 0
        for i in range(length):
            upper = lower + 100
            if upper > length:
                upper = length
            indices = lrange[lower:upper]
            DataSlice = self.rr1.indexselectdata(Data, indices)
            invariant[lower:upper] = self.orderinvariantSlice(DataSlice)
            lower = upper
            if lower >= length:
                break
        return invariant

    def printtestinvariant(self, length, invariant):
        #
        a = self.alpha
        #
        assert length > 0
        assert a > 1
        #
        invariant_sorted, indices = torch.sort(invariant, 1)
        #
        invariant_pre = invariant_sorted[:, 0 : a - 1]
        invariant_post = invariant_sorted[:, 1:a]
        invariant_delta = invariant_post - invariant_pre
        #
        assert ((invariant_delta >= 0).all(1)).all(0)
        delta_avg = (invariant_delta.sum(1)).sum(0).to(torch.float) / (
            itf(length) * itf(a - 1)
        )
        #
        delta_sum = (invariant_delta == 0).to(torch.int64).sum(1)
        print(
            "invariant delta average is",
            numpr(delta_avg, 1),
            "for length",
            itp(length),
        )
        for k in range(a - 1):
            locnumber = (delta_sum == k).to(torch.int64).sum(0)
            print(
                "number of locations with delta sum",
                itp(k),
                "is",
                itp(locnumber),
            )
        return

    def matrixinvariant(self, Data, ivector, gvector):
        #
        a = self.alpha
        #
        length = Data["length"]
        #
        eq, iz = self.geteq(Data)
        #
        vlength = len(ivector)
        assert len(gvector) == vlength
        #
        irangevx = ivector.view(vlength, 1, 1, 1, 1).expand(
            vlength, a, a, a, a
        )
        grangevx = gvector.view(vlength, 1, 1, 1, 1).expand(
            vlength, a, a, a, a
        )
        xrangevx = arangeic(a).view(1, a, 1, 1, 1).expand(vlength, a, a, a, a)
        yrangevx = arangeic(a).view(1, 1, a, 1, 1).expand(vlength, a, a, a, a)
        zrangevx = arangeic(a).view(1, 1, 1, a, 1).expand(vlength, a, a, a, a)
        wrangevx = arangeic(a).view(1, 1, 1, 1, a).expand(vlength, a, a, a, a)
        #
        xtransform = self.sga.grouptable[grangevx, xrangevx]
        ytransform = self.sga.grouptable[grangevx, yrangevx]
        ztransform = self.sga.grouptable[grangevx, zrangevx]
        wtransform = self.sga.grouptable[grangevx, wrangevx]
        #
        eq_transform = eq[
            irangevx, xtransform, ytransform, ztransform, wtransform
        ]
        #
        matrixvx = self.matrix.view(1, a, a, a, a).expand(vlength, a, a, a, a)
        matrix_invariant = (
            ((eq_transform.to(torch.int64)) * matrixvx)
            .view(vlength, a * a * a * a)
            .sum(1)
        )
        #
        return matrix_invariant

    def to_eqfunction(self, length, eq, iz):
        #
        a = self.alpha
        a2 = a * a
        #
        eqview = eq.view(length * a2, a2)
        izview = iz.view(length * a2)
        #
        numerical = arangeic(a2).view(1, a2).expand(length * a2, a2) + 1
        numerical_eq = numerical * (eqview.to(torch.int64))
        #
        values, eqfunctionv = torch.max(numerical_eq, 1)
        eqfunctionv[izview] = a2
        eqfunction = eqfunctionv.view(length, a2)
        return eqfunction

    def from_eqfunction(self, length, eqfunction):
        #
        #
        a = self.alpha
        a2 = a * a
        #
        eqf1vx = eqfunction.view(length, a2, 1).expand(length, a2, a2)
        eqf2vx = eqfunction.view(length, 1, a2).expand(length, a2, a2)
        #
        eq = eqf1vx == eqf2vx
        #
        iz = eqfunction.view(length, a2) == a2
        #
        return eq, iz

    def transform_eqfunction(self, length, eq, iz, gvector):
        #
        a = self.alpha
        a2 = a * a
        #
        assert len(gvector) == length
        #
        eqv = eq.view(length, a, a, a, a)
        izv = iz.view(length, a, a)
        #
        irangevx = (
            arangeic(length)
            .view(length, 1, 1, 1, 1)
            .expand(length, a, a, a, a)
        )
        grangevx = gvector.view(length, 1, 1, 1, 1).expand(length, a, a, a, a)
        xrangevx = arangeic(a).view(1, a, 1, 1, 1).expand(length, a, a, a, a)
        yrangevx = arangeic(a).view(1, 1, a, 1, 1).expand(length, a, a, a, a)
        zrangevx = arangeic(a).view(1, 1, 1, a, 1).expand(length, a, a, a, a)
        wrangevx = arangeic(a).view(1, 1, 1, 1, a).expand(length, a, a, a, a)
        #
        xtransform = self.sga.grouptable[grangevx, xrangevx]
        ytransform = self.sga.grouptable[grangevx, yrangevx]
        ztransform = self.sga.grouptable[grangevx, zrangevx]
        wtransform = self.sga.grouptable[grangevx, wrangevx]
        #
        eq_transform = eqv[
            irangevx, xtransform, ytransform, ztransform, wtransform
        ].view(length, a2, a2)
        #
        irangeZvx = arangeic(length).view(length, 1, 1).expand(length, a, a)
        grangeZvx = gvector.view(length, 1, 1).expand(length, a, a)
        xrangeZvx = arangeic(a).view(1, a, 1).expand(length, a, a)
        yrangeZvx = arangeic(a).view(1, 1, a).expand(length, a, a)
        #
        xtransformZ = self.sga.grouptable[grangeZvx, xrangeZvx]
        ytransformZ = self.sga.grouptable[grangeZvx, yrangeZvx]
        #
        iz_transform = izv[irangeZvx, xtransformZ, ytransformZ].view(
            length, a2
        )
        #
        eqfunction_transform = self.to_eqfunction(
            length, eq_transform, iz_transform
        )
        return eqfunction_transform

    def data_eqfunction_transform(self, Data, ivector, gvector):
        #
        a = self.alpha
        a2 = a * a
        #
        DataVector = self.rr1.indexselectdata(Data, ivector)
        length = DataVector["length"]
        #
        eq, iz = self.geteq(DataVector)
        #
        assert len(gvector) == length
        #
        # eqfunction = self.to_eqfunction(length,eq,iz)
        #
        eqfunction_transform = self.transform_eqfunction(
            length, eq, iz, gvector
        )
        #
        return eqfunction_transform

    def uniqueinstances(self, length, eq_function):
        #
        a = self.alpha
        a2 = a * a
        #
        assert length > 0
        #
        eqf1vx = eq_function.view(length, 1, a2).expand(length, length, a2)
        eqf2vx = eq_function.view(1, length, a2).expand(length, length, a2)
        #
        equivalent = (eqf1vx == eqf2vx).all(2)
        #
        first = arangeic(length).view(length, 1).expand(length, length)
        second = arangeic(length).view(1, length).expand(length, length)
        #
        isrep = ((~equivalent) | (first <= second)).all(1)
        #
        unique_length = isrep.to(torch.int64).sum(0)
        eq_unique = eq_function[isrep]
        #
        assert unique_length > 0
        #
        return unique_length, eq_unique

    def addSlice(self, length, eq_function):
        #
        a = self.alpha
        a2 = a * a
        #
        ulength, eq_unique = self.uniqueinstances(length, eq_function)
        #
        alength = self.eqlength
        #
        if alength == 0:
            self.eqlist = eq_unique
            self.eqlength = ulength
            return
        alreadyvx = self.eqlist.view(1, alength, a2).expand(
            ulength, alength, a2
        )
        newvx = eq_unique.view(ulength, 1, a2).expand(ulength, alength, a2)
        #
        already_known = ((alreadyvx == newvx).all(2)).any(1)
        #
        detection = ~already_known
        detected_length = detection.to(torch.int64).sum(0)
        eq_detected = eq_unique[detection]
        #
        #
        self.eqlist = torch.cat((self.eqlist, eq_detected), 0)
        #
        self.eqlength += detected_length
        #
        return

    def addinstances(self, length, eq_function):
        #
        if length == 0:
            # print("no instances to add")
            return
        #
        lower = 0
        for i in range(length):
            upper = lower + 500
            if upper > length:
                upper = length
            length_slice = upper - lower
            eq_slice = eq_function[lower:upper]
            self.addSlice(length_slice, eq_slice)
            lower = upper
            if lower >= length:
                break
        return

    def processBasic(self, Data):
        #
        a = self.alpha
        assert a > 1
        #
        length = Data["length"]
        print(
            "current eqlength",
            itp(self.eqlength),
            "processing data of length",
            itp(length),
        )
        #
        eq, iz = self.geteq(Data)
        #
        eq_function = self.to_eqfunction(length, eq, iz)
        #
        self.addinstances(length, eq_function)
        #
        return

    def process(self, Data):
        #
        a = self.alpha
        assert a > 1
        #
        length = Data["length"]
        gl = self.sga.gtlength
        #
        invariant = self.orderinvariant(Data)
        #
        lrangevxr = (
            arangeic(length)
            .view(length, 1, 1)
            .expand(length, gl, a)
            .reshape(length * gl, a)
        )
        gtvxr = (
            self.sga.grouptable.view(1, gl, a)
            .expand(length, gl, a)
            .reshape(length * gl, a)
        )
        #
        invariant_gt = invariant[lrangevxr, gtvxr]
        detection = (
            (invariant_gt[:, 0 : a - 1]) <= (invariant_gt[:, 1:a])
        ).all(1)
        dlength = detection.to(torch.int64).sum(0)
        ivector = (
            arangeic(length)
            .view(length, 1)
            .expand(length, gl)
            .reshape(length * gl)
        )[detection]
        gvector = (
            arangeic(gl).view(1, gl).expand(length, gl).reshape(length * gl)
        )[detection]
        #
        verification = torch.zeros((length), dtype=torch.bool, device=Dvc)
        verification[ivector] = True
        #
        assert verification.all(0)
        #
        eq_function = self.data_eqfunction_transform(Data, ivector, gvector)
        #
        self.addinstances(dlength, eq_function)
        #
        return

    def checklocationSlice(self, length, eq_function):
        #
        a = self.alpha
        a2 = a * a
        #
        location = torch.zeros((length), dtype=torch.int64, device=Dvc)
        location[:] = -1
        #
        alength = self.eqlength
        #
        predetect_already = (self.eqlist.view(alength, a2)).sum(1)
        predetect_current = (eq_function.view(length, a2)).sum(1)
        predetect_already_vxr = (
            predetect_already.view(alength, 1)
            .expand(alength, length)
            .reshape(alength * length)
        )
        predetect_current_vxr = (
            predetect_current.view(1, length)
            .expand(alength, length)
            .reshape(alength * length)
        )
        #
        predetection = predetect_already_vxr == predetect_current_vxr
        #
        alrangevxr = (
            arangeic(alength)
            .view(alength, 1)
            .expand(alength, length)
            .reshape(alength * length)
        )
        lrangevxr = (
            arangeic(length)
            .view(1, length)
            .expand(alength, length)
            .reshape(alength * length)
        )
        #
        alrvector = alrangevxr[predetection]
        currvector = lrangevxr[predetection]
        #
        eq_already = (self.eqlist.view(alength, a2))[alrvector]
        eq_current = (eq_function.view(length, a2))[currvector]
        #
        detection = (eq_already == eq_current).all(1)
        #
        already_detected = alrvector[detection]
        current_detected = currvector[detection]
        #
        location[current_detected] = already_detected
        #
        return location

    def checklocation(self, length, eq_function):
        #
        a = self.alpha
        a2 = a * a
        #
        location = torch.zeros((length), dtype=torch.int64, device=Dvc)
        #
        lower = 0
        for i in range(length):
            upper = lower + 100
            if upper > length:
                upper = length
            length_slice = upper - lower
            eq_slice = eq_function[lower:upper]
            location[lower:upper] = self.checklocationSlice(
                length_slice, eq_slice
            )
            lower = upper
            if lower >= length:
                break
        return location

    def highestlocation(self, length, eq_function):
        #
        a = self.alpha
        a2 = a * a
        #
        gl = self.sga.gtlength
        #
        eq_vxr = (
            eq_function.view(length, 1, a2)
            .expand(length, gl, a2)
            .reshape(length * gl, a2)
        )
        gvector = (
            arangeic(gl).view(1, gl).expand(length, gl).reshape(length * gl)
        )
        grouplength = length * gl
        #
        eq, iz = self.from_eqfunction(grouplength, eq_vxr)
        eq_transform = self.transform_eqfunction(grouplength, eq, iz, gvector)
        #
        location = self.checklocation(grouplength, eq_transform)
        locationv = location.view(length, gl)
        highest, indices = torch.max(locationv, 1)
        return highest

    def sieve(self):
        #
        alength = self.eqlength
        print("doing sieve on", itp(alength), "locations")
        #
        highest_location = torch.zeros(
            (alength), dtype=torch.int64, device=Dvc
        )
        #
        lower = 0
        for i in range(alength):
            print("---", lower)
            upper = lower + 100
            if upper > alength:
                upper = alength
            eq_slice = self.eqlist[lower:upper]
            length_slice = upper - lower
            #
            highest_location[lower:upper] = self.highestlocation(
                length_slice, eq_slice
            )
            lower = upper
            if lower >= alength:
                break
        #
        negative_detect = highest_location < 0
        negative_count = negative_detect.to(torch.int64).sum(0)
        print(
            "unfortunate locations with negative values :", itp(negative_count)
        )
        alrange = arangeic(alength)
        good_detect = highest_location <= alrange
        good_count = good_detect.to(torch.int64).sum(0)
        print("detected", itp(good_count), "good locations")
        print("finished sieve")
        return
