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

from relations_1 import Relations1
from utils import arangeic, composedetections


class Relations2:
    def __init__(self, pp):
        #
        self.pp = pp
        #
        self.rr1 = Relations1(pp)
        #
        self.alpha = self.pp.alpha
        self.alpha2 = self.alpha * self.alpha
        self.alpha2z = self.alpha2 + 1
        self.alpha3 = self.alpha * self.alpha * self.alpha
        self.alpha3z = self.alpha3 + 1
        self.beta = self.pp.beta
        self.betaz = self.beta + 1
        #
        #
        self.halfones_count = 0
        self.impossible_basic_count = 0

    # new steps with prod, left, right, ternary

    def modifyternaryStep(self, Data):
        a = self.alpha
        bz = self.betaz
        length = Data["length"]
        prod = Data["prod"]
        left = Data["left"]
        right = Data["right"]
        ternary = Data["ternary"]
        ivx = (
            arangeic(length)
            .view(length, 1, 1, 1, 1)
            .expand(length, a, a, a, bz)
        )
        xvx = arangeic(a).view(1, a, 1, 1, 1).expand(length, a, a, a, bz)
        yvx = arangeic(a).view(1, 1, a, 1, 1).expand(length, a, a, a, bz)
        zvx = arangeic(a).view(1, 1, 1, a, 1).expand(length, a, a, a, bz)
        pvx = arangeic(bz).view(1, 1, 1, 1, bz).expand(length, a, a, a, bz)
        #
        nter0_left = (prod[ivx, yvx, zvx, pvx] & left[ivx, xvx, pvx, 0]).any(4)
        nter1_left = (prod[ivx, yvx, zvx, pvx] & left[ivx, xvx, pvx, 1]).any(4)
        #
        nter0_right = (prod[ivx, xvx, yvx, pvx] & right[ivx, pvx, zvx, 0]).any(
            4
        )
        nter1_right = (prod[ivx, xvx, yvx, pvx] & right[ivx, pvx, zvx, 1]).any(
            4
        )
        #
        nter0v = nter0_left & nter0_right
        nter1v = nter1_left & nter1_right
        #
        newternary = ternary.clone()
        newternary[:, :, :, :, 0] = ternary[:, :, :, :, 0] & nter0v
        newternary[:, :, :, :, 1] = ternary[:, :, :, :, 1] & nter1v
        #
        NewData = self.rr1.copydata(Data)
        NewData["ternary"] = newternary.detach()
        #
        return NewData

    def modifyleftrightStep(self, Data):
        a = self.alpha
        bz = self.betaz
        length = Data["length"]
        prod = Data["prod"]
        left = Data["left"]
        right = Data["right"]
        ternary = Data["ternary"]
        #
        prodstats = prod.to(torch.int64).sum(3)
        unique = prodstats == 1
        #
        ivx = (
            arangeic(length)
            .view(length, 1, 1, 1, 1)
            .expand(length, a, a, a, bz)
        )
        xvx = arangeic(a).view(1, a, 1, 1, 1).expand(length, a, a, a, bz)
        yvx = arangeic(a).view(1, 1, a, 1, 1).expand(length, a, a, a, bz)
        zvx = arangeic(a).view(1, 1, 1, a, 1).expand(length, a, a, a, bz)
        pvx = arangeic(bz).view(1, 1, 1, 1, bz).expand(length, a, a, a, bz)
        #
        nleft0 = (
            (
                (~prod[ivx, yvx, zvx, pvx])
                | (~unique[ivx, yvx, zvx])
                | ternary[ivx, xvx, yvx, zvx, 0]
            ).all(3)
        ).all(2)
        nleft1 = (
            (
                (~prod[ivx, yvx, zvx, pvx])
                | (~unique[ivx, yvx, zvx])
                | ternary[ivx, xvx, yvx, zvx, 1]
            ).all(3)
        ).all(2)
        #
        nright0 = (
            (
                (~prod[ivx, xvx, yvx, pvx])
                | (~unique[ivx, xvx, yvx])
                | ternary[ivx, xvx, yvx, zvx, 0]
            ).all(2)
        ).all(1)
        nright1 = (
            (
                (~prod[ivx, xvx, yvx, pvx])
                | (~unique[ivx, xvx, yvx])
                | ternary[ivx, xvx, yvx, zvx, 1]
            ).all(2)
        ).all(1)
        #
        newleft = left.clone()
        newright = right.clone()
        #
        newleft[:, :, :, 0] = left[:, :, :, 0] & nleft0
        newleft[:, :, :, 1] = left[:, :, :, 1] & nleft1
        newright[:, :, :, 0] = right[:, :, :, 0] & (nright0.permute(0, 2, 1))
        newright[:, :, :, 1] = right[:, :, :, 1] & (nright1.permute(0, 2, 1))
        #
        NewData = self.rr1.copydata(Data)
        NewData["left"] = newleft.detach()
        NewData["right"] = newright.detach()
        #
        return NewData

    def modifyprodStep(self, Data):
        a = self.alpha
        bz = self.betaz
        length = Data["length"]
        prod = Data["prod"]
        left = Data["left"]
        right = Data["right"]
        ternary = Data["ternary"]
        #
        lvx = (
            arangeic(length)
            .view(length, 1, 1, 1, 1)
            .expand(length, a, a, a, bz)
        )
        xvx = arangeic(a).view(1, a, 1, 1, 1).expand(length, a, a, a, bz)
        yvx = arangeic(a).view(1, 1, a, 1, 1).expand(length, a, a, a, bz)
        zvx = arangeic(a).view(1, 1, 1, a, 1).expand(length, a, a, a, bz)
        pvx = arangeic(bz).view(1, 1, 1, 1, bz).expand(length, a, a, a, bz)
        #
        leftbin01 = left[lvx, xvx, pvx, 0] | ternary[lvx, xvx, yvx, zvx, 1]
        leftbin10 = left[lvx, xvx, pvx, 1] | ternary[lvx, xvx, yvx, zvx, 0]
        #
        rightbin01 = right[lvx, pvx, zvx, 0] | ternary[lvx, xvx, yvx, zvx, 1]
        rightbin10 = right[lvx, pvx, zvx, 1] | ternary[lvx, xvx, yvx, zvx, 0]
        #
        newprod = prod.clone()
        newprod = newprod & ((leftbin01 & leftbin10).all(1))
        newprod = newprod & ((rightbin01 & rightbin10).all(3))
        #
        NewData = self.rr1.copydata(Data)
        NewData["prod"] = newprod.detach()
        #
        return NewData

    def process(self, Data):
        length = Data["length"]
        if length == 0:
            return Data
        #
        #
        OutputData = self.rr1.copydata(Data)
        nprod = Data["prod"]
        nprodstats = nprod.to(torch.int64).sum(3)
        subset = ((nprodstats > 0).all(2)).all(1)
        NextData = self.rr1.detectsubdata(Data, subset)
        if subset.to(torch.int).sum(0) == 0:
            return OutputData
        for _ in range(1000):
            priorknowledge = self.rr1.knowledge(NextData)
            NextData = self.modifyternaryStep(NextData)
            NextData = self.modifyleftrightStep(NextData)
            NextData = self.modifyprodStep(NextData)
            nextknowledge = self.rr1.knowledge(NextData)
            nextdonedetect = priorknowledge >= nextknowledge
            subset_nextdone = composedetections(length, subset, nextdonedetect)
            NextDoneData = self.rr1.detectsubdata(NextData, nextdonedetect)
            OutputData = self.rr1.insertdata(
                OutputData, subset_nextdone, NextDoneData
            )
            subset = subset & (~subset_nextdone)
            if subset.to(torch.int).sum(0) == 0:
                break
            NextData = self.rr1.detectsubdata(NextData, ~nextdonedetect)
        return OutputData

    def impossibleFilter(self, Data):
        a = self.alpha
        a3 = self.alpha3
        bz = self.betaz
        length = Data["length"]
        prod = Data["prod"]
        left = Data["left"]
        right = Data["right"]
        ternary = Data["ternary"]
        #
        prodstats = prod.to(torch.int64).sum(3)
        possible = ((prodstats > 0).all(2)).all(1)
        #
        leftv = left.view(length, a * bz, 2)
        rightv = right.view(length, bz * a, 2)
        ternaryv = ternary.view(length, a3, 2)
        left_possible = (leftv.any(2)).all(1)
        right_possible = (rightv.any(2)).all(1)
        ternary_possible = (ternaryv.any(2)).all(1)
        #
        detection = (
            (~possible)
            | (~left_possible)
            | (~right_possible)
            | (~ternary_possible)
        )
        return detection

    def profileFilter(self, Data):
        a = self.alpha
        bz = self.betaz
        length = Data["length"]
        left = Data["left"]
        right = Data["right"]
        left_def = (left.to(torch.int64).sum(3)) == 1
        right_def = (right.to(torch.int64).sum(3)) == 1
        profile_def = (left_def.all(1)) & (right_def.all(2))
        left_pro = (left[:, :, :, 0]).permute(0, 2, 1)
        right_pro = right[:, :, :, 0]
        profile = torch.cat((left_pro, right_pro), 2)
        profile_vx1 = profile.view(length, bz, 1, 2 * a).expand(
            length, bz, bz, 2 * a
        )
        profile_vx2 = profile.view(length, 1, bz, 2 * a).expand(
            length, bz, bz, 2 * a
        )
        same_profile = (profile_vx1 == profile_vx2).all(3)
        #
        profile_def1 = profile_def.view(length, bz, 1).expand(length, bz, bz)
        profile_def2 = profile_def.view(length, 1, bz).expand(length, bz, bz)
        #
        same_profile_def = profile_def1 & profile_def2 & same_profile
        prange1 = arangeic(bz).view(1, bz, 1).expand(length, bz, bz)
        prange2 = arangeic(bz).view(1, 1, bz).expand(length, bz, bz)
        #
        detection = (((prange1 != prange2) & same_profile_def).any(2)).any(1)
        #
        return detection

    def doneFilter(self, Data):
        prod = Data["prod"]
        prodstats = prod.to(torch.int64).sum(3)
        return ((prodstats == 1).all(2)).all(1)

    def halfonesFilter(self, Data):  # only look at cases where the number of
        # ones on the left is >= the number on the right
        left = Data["left"]
        right = Data["right"]
        leftstats = left.to(torch.int64).sum(3)
        rightstats = right.to(torch.int64).sum(3)
        #
        assert (((leftstats <= 1).all(2)).all(1)).all(0)
        #
        leftones = (left[:, :, :, 1].to(torch.int64).sum(2)).sum(1)
        right_isone = (rightstats == 1) & right[:, :, :, 1]
        rightones = (right_isone.to(torch.int64).sum(2)).sum(1)
        #
        #
        detection = rightones > leftones
        #
        return detection

    def filterdata(self, Data):  #
        #
        #
        impossibledetect = self.impossibleFilter(Data)
        profiledetect = self.profileFilter(Data)
        #
        if self.pp.profile_filter_on:
            impossibledetect = impossibledetect | profiledetect
        #
        self.impossible_basic_count += impossibledetect.to(torch.int64).sum(0)
        # experimental:
        if self.pp.halfones_filter_on:
            halfonesdetect = self.halfonesFilter(Data)
            self.halfones_count += (
                (halfonesdetect & (~impossibledetect)).to(torch.int64).sum(0)
            )
            impossibledetect = impossibledetect | halfonesdetect
        #
        donedetect = self.doneFilter(Data)
        donedetect = donedetect & (~impossibledetect)
        #
        activedetect = (~impossibledetect) & ~donedetect
        #
        return activedetect, donedetect, impossibledetect
