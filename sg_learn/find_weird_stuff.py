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
# next: a class that can be used to search for specific examples, it was a debugging tool kept here for reference
import torch

from constants import Dvc
from utils import arangeic, itp, nump


class FindWeirdStuff:
    def __init__(self, Dd, Mm):
        #
        #
        self.Dd = Dd
        self.Ll = self.Dd.Ll
        self.Mm = Mm
        #
        self.Pp = self.Dd.Pp
        #
        self.rr4 = self.Dd.rr4
        self.rr3 = self.rr4.rr3
        self.rr2 = self.rr4.rr2
        self.rr1 = self.rr4.rr1
        self.alpha = self.Pp.alpha
        self.alpha2 = self.Pp.alpha2
        self.alpha3 = self.Pp.alpha3
        self.alpha3z = self.Pp.alpha3z
        self.beta = self.Pp.beta
        self.betaz = self.Pp.betaz
        #

    def sample_box(self, xlower, xupper, ylower, yupper):
        #
        smb, DataBatch, scorebatch = self.Ll.selectminibatch(500)
        #
        if not smb:
            print("select mini batch fails")
            return False, None
        predictedscore = self.Mm.network(DataBatch)
        #
        detectionx = (xlower < scorebatch) & (scorebatch < xupper)
        detectiony = (ylower < predictedscore) & (predictedscore < yupper)
        detection = detectionx & detectiony
        #
        DetectedData = self.rr1.detectsubdata(DataBatch, detection)
        #
        return True, DetectedData

    def printright(self, right, loc):
        #
        a = self.alpha
        a2 = a * a
        a3 = a * a * a
        a3z = a3 + 1
        b = self.beta
        bz = b + 1
        #
        prarray = torch.zeros((bz, a), dtype=torch.int64, device=Dvc)
        for x in range(bz):
            for y in range(a):
                column = right[loc, x, y]
                prarray[x, y] = self.Dd.printcolumn2(column)
        print(nump(prarray))
        return

    def print_prod_left_right(self, Data, i):
        prod = Data["prod"]
        left = Data["left"]
        right = Data["right"]
        #
        print("---------------------------------------------------")
        print(
            "at location", itp(i), "the prod, left and right are respectively"
        )
        self.Dd.printprod(prod, i)
        self.Dd.printleft(left, i)
        self.printright(right, i)
        print("---------------------------------------------------")
        return

    def print_column_bool(self, column):
        clength = len(column)
        toprint = 3 * (10 ** (clength))
        for q in range(clength):
            toprint += column[q].to(torch.int64) * (10 ** q)
        return toprint

    def print_bool_tensor(self, a, b, c, btensor):
        prarray = torch.zeros((a, b), dtype=torch.int64, device=Dvc)
        for x in range(a):
            for y in range(b):
                column = btensor[x, y]
                prarray[x, y] = self.print_column_bool(column)
        print(nump(prarray))
        return

    def print_one_sample_from_box(self, xlower, xupper, ylower, yupper):
        #
        sb, DetectedData = self.sample_box(xlower, xupper, ylower, yupper)
        #
        if not sb:
            print("exit from one sample box")
            return
        length = DetectedData["length"]
        if length == 0:
            print("didn't detect any samples in this box")
            return
        #
        AssocData = self.rr2.process(DetectedData)
        activedetect, donedetect, impossibledetect = self.rr2.filterdata(
            AssocData
        )
        permutation = torch.randperm(length, device=Dvc)
        loc = permutation[0]
        print("before processing:")
        self.print_prod_left_right(DetectedData, loc)
        print("after processing:")
        self.print_prod_left_right(AssocData, loc)
        print("the status of this location is :", end=" ")
        if activedetect[loc]:
            print("active")
        if donedetect[loc]:
            print("done")
        if impossibledetect[loc]:
            print("impossible")
        return

    def av_root(self, DataToSplit, i):
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
        availablexypi = self.rr1.availablexyp(length, prod).view(
            length, a, a, bz
        )[i]
        #
        print("at root, availablexyp is:")
        self.print_bool_tensor(a, a, bz, availablexypi)
        #
        return

    def split_by_hand(self, DataToSplit, i, x, y, p):
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
        availablexypi = self.rr1.availablexyp(length, prod).view(
            length, a, a, bz
        )[i]
        #
        available_instance = availablexypi[x, y, p]
        if available_instance:
            print("this instance", itp(x), itp(y), itp(p), "is available")
        else:
            print("this instance", itp(x), itp(y), itp(p), "is not available")
            return
        #
        ivector = torch.zeros((1), dtype=torch.int64, device=Dvc)
        ivector[:] = i
        xvector = torch.zeros((1), dtype=torch.int64, device=Dvc)
        xvector[:] = x
        yvector = torch.zeros((1), dtype=torch.int64, device=Dvc)
        yvector[:] = y
        pvector = torch.zeros((1), dtype=torch.int64, device=Dvc)
        pvector[:] = p
        #
        #
        NewData = self.rr1.upsplitting(
            DataToSplit, ivector, xvector, yvector, pvector
        )
        #
        #
        ndlength = NewData["length"]
        assert ndlength == 1
        #
        AssocNewData = self.rr2.process(NewData)
        #
        newactive, newdone, newimpossible = self.rr2.filterdata(AssocNewData)
        if newactive[0]:
            print("active")
        if newdone[0]:
            print("done")
        if newimpossible[0]:
            print("impossible")
        #
        AND_prod = AssocNewData["prod"]
        AND_length = 1
        AND_availablexyp = self.rr1.availablexyp(AND_length, AND_prod).view(
            1, a, a, bz
        )[0]
        print("after cut at", itp(x), itp(y), itp(p), "availablexyp is:")
        self.print_bool_tensor(a, a, bz, AND_availablexyp)
        print("prod is")
        self.print_bool_tensor(a, a, bz, AND_prod[0])
        #
        #
        return AND_prod

    def show_cut_column(
        self, sigma, x, y
    ):  # shows the results of (x,y,p) cut starting from root, for all p
        #
        instancevector, trainingvector, title_text = self.Dd.InOne(sigma)
        InitialData = self.Dd.initialdata(instancevector, 0)
        #
        self.av_root(InitialData, 0)
        #
        for p in range(self.betaz):
            #
            andprod = self.split_by_hand(InitialData, 0, x, y, p)
        return

    def searchprod(self, Data, trprod):
        #
        a = self.alpha
        bz = self.betaz
        #
        length = Data["length"]
        prod = Data["prod"]
        #
        if length == 0:
            print("length is 0")
            return 0, None
        prodv = prod.view(length, a * a * bz)
        trprodv = trprod.view(1, a * a * bz).expand(length, a * a * bz)
        #
        detection = (prodv == trprodv).all(1)
        #
        detected_indices = arangeic(length)[detection]
        detected_length = detection.to(torch.int64).sum(0)
        print(
            "detected", itp(detected_length), "occurences out of", itp(length)
        )
        return detected_length, detected_indices

    def tracer(self, trprod):
        #
        print("Examples:", end=" ")
        dl, di = self.searchprod(self.Ll.Examples, trprod)
        print("ExamplesPrePool:", end=" ")
        dl, di = self.searchprod(self.Ll.ExamplesPrePool, trprod)
        print("ExplorePrePool:", end=" ")
        dl, di = self.searchprod(self.Ll.ExplorePrePool, trprod)
        print("OutlierPrePool:", end=" ")
        dl, di = self.searchprod(self.Ll.OutlierPrePool, trprod)
        #
        #
        return

    def tracer_root(self, sigma):
        #
        instancevector, trainingvector, title_text = self.Dd.InOne(sigma)
        InitialData = self.Dd.initialdata(instancevector, 0)
        #
        trprod = InitialData["prod"][0]
        #
        self.tracer(trprod)
        #
        return

    def tracer_subroot(self, sigma, x, y, p):
        #
        instancevector, trainingvector, title_text = self.Dd.InOne(sigma)
        InitialData = self.Dd.initialdata(instancevector, 0)
        #
        trprod = self.split_by_hand(InitialData, 0, x, y, p)
        #
        self.tracer(trprod)
