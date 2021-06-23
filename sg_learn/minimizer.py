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
from driver import Driver
from historical import Historical
from relations_4 import Relations4
from utils import CoherenceError, arangeic, itp, itt, nump

# next: the class used to prove the theoretical minimum (this is for alpha,beta = 3,2)
# note that the parameters and model should be initialized for (3,2)


class Minimizer:  # this becomes the first element of the relations datatype
    def __init__(self, model, sigma, cutx, cuty, cutp, HST: Historical):
        #
        #
        self.Mm = model
        self.Pp = self.Mm.pp
        self.Dd = Driver(self.Pp, HST)
        assert self.Pp.alpha == 3
        assert self.Pp.beta == 2
        #
        self.sigma = sigma
        #
        print("Minimizer for sigma =", sigma)
        #
        self.rr4 = Relations4(self.Pp, HST)
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
        self.length_max = 500000
        #
        instancevector, _, _ = self.Dd.InOne(self.sigma)
        self.InitialData = self.Dd.initialdata(instancevector, 0)
        #
        self.CurrentData = self.rr1.nulldata()
        self.FullData = self.rr1.nulldata()
        self.FullDoneData = self.rr1.nulldata()
        #
        self.up = torch.zeros((self.length_max), dtype=torch.int64, device=Dvc)
        self.xvalue = torch.zeros(
            (self.length_max), dtype=torch.int64, device=Dvc
        )
        self.yvalue = torch.zeros(
            (self.length_max), dtype=torch.int64, device=Dvc
        )
        self.pvalue = torch.zeros(
            (self.length_max), dtype=torch.int64, device=Dvc
        )
        #
        self.lowerbound = torch.zeros(
            (self.length_max), dtype=torch.int64, device=Dvc
        )
        self.upperbound = torch.zeros(
            (self.length_max), dtype=torch.int64, device=Dvc
        )
        #
        self.down = torch.zeros(
            (self.length_max, self.alpha, self.alpha, self.betaz),
            dtype=torch.int64,
            device=Dvc,
        )
        self.down[:, :, :, :] = -2
        #
        self.split = torch.zeros(
            (self.length_max), dtype=torch.bool, device=Dvc
        )
        self.inplay = torch.zeros(
            (self.length_max), dtype=torch.bool, device=Dvc
        )
        #
        self.availablexyp = torch.zeros(
            (self.length_max, self.alpha, self.alpha, self.betaz),
            dtype=torch.bool,
            device=Dvc,
        )
        self.availablexy = torch.zeros(
            (self.length_max, self.alpha, self.alpha),
            dtype=torch.bool,
            device=Dvc,
        )
        #
        self.cutx = cutx
        self.cuty = cuty
        self.cutp = cutp
        print("cut to check is x y p =", cutx, cuty, cutp)
        #
        self.combo_all()

    def next_stage_data(self, DataToSplit):
        a = self.alpha
        a2 = self.alpha2
        bz = self.betaz
        length = DataToSplit["length"]
        prod = DataToSplit["prod"]
        availablexyp = self.rr1.availablexyp(length, prod).view(length, a2, bz)
        avxyp_denom = itt(length).to(torch.float)
        if avxyp_denom < 0.1:
            avxyp_denom = 1.0
        lrangevxr = (
            arangeic(length)
            .view(length, 1, 1)
            .expand(length, a2, bz)
            .reshape(length * a2 * bz)
        )
        xyvectorvxr = (
            arangeic(a2)
            .view(1, a2, 1)
            .expand(length, a2, bz)
            .reshape(length * a2 * bz)
        )
        bzrangevxr = (
            arangeic(bz)
            .view(1, 1, bz)
            .expand(length, a2, bz)
            .reshape(length * a2 * bz)
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
        for _ in range(ndlength):
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
        assert len(newactive) == len(ivector_vert)
        assert newactive.to(torch.int64).sum(0) == NewActiveData["length"]
        #
        return (
            ivector_vert,
            xvector_vert,
            yvector_vert,
            pvector_vert,
            newactive,
            newdone,
            newimpossible,
            NewActiveData,
            NewDoneData,
        )

    def manage_initial_stage(self):
        #
        self.FullData = self.rr1.copydata(self.InitialData)
        #
        fdlength = self.FullData["length"]
        assert fdlength == 1
        #
        #
        self.up[0] = -1
        self.xvalue[0] = -1
        self.yvalue[0] = -1
        self.pvalue[0] = -1
        #
        self.lowerbound[0] = 1
        self.upperbound[0] = 1000
        #
        #
        self.split[0] = False
        self.inplay[0] = True
        #
        newactive_length = self.InitialData["length"]
        newactive_prod = self.InitialData["prod"]
        #
        new_availablexyp = self.rr1.availablexyp(
            newactive_length, newactive_prod
        ).view(newactive_length, self.alpha, self.alpha, self.betaz)
        new_availablexy = new_availablexyp.any(3)
        #
        self.availablexyp[0] = new_availablexyp
        self.availablexy[0] = new_availablexy
        #
        new_fdlength = self.FullData["length"]
        #
        self.FullData["info"][:, self.Pp.fulldata_location] = arangeic(
            new_fdlength
        )

    def manage_next_stage(self):
        #
        fdlength = self.FullData["length"]
        if fdlength == 0:
            self.manage_initial_stage()
            return
        #
        fd_split = self.split[0:fdlength]
        fd_inplay = self.inplay[0:fdlength]
        #
        fd_current = fd_inplay & ~fd_split
        DataToSplit = self.rr1.detectsubdata(self.FullData, fd_current)
        #
        fd_index = arangeic(fdlength)[fd_current]
        #
        (
            ivector_vert,
            xvector_vert,
            yvector_vert,
            pvector_vert,
            newactive,
            newdone,
            newimpossible,
            NewActiveData,
            NewDoneData,
        ) = self.next_stage_data(DataToSplit)
        #
        inew = fd_index[ivector_vert[newactive]]
        xnew = xvector_vert[newactive]
        ynew = yvector_vert[newactive]
        pnew = pvector_vert[newactive]
        #
        newactive_length = NewActiveData["length"]
        newactive_prod = NewActiveData["prod"]
        #
        lower = itt(fdlength).clone()
        upper = fdlength + newactive_length
        if upper > self.length_max:
            print(
                "upper is",
                itp(upper),
                "whereas max length is",
                itp(self.length_max),
            )
            raise CoherenceError("length overflow")
        self.FullData = self.rr1.appenddata(self.FullData, NewActiveData)
        self.FullDoneData = self.rr1.appenddata(self.FullDoneData, NewDoneData)
        #
        self.up[lower:upper] = inew
        self.xvalue[lower:upper] = xnew
        self.yvalue[lower:upper] = ynew
        self.pvalue[lower:upper] = pnew
        #
        self.lowerbound[lower:upper] = 1
        self.upperbound[lower:upper] = 1000
        #
        assert (~self.split[inew]).all(0)
        assert self.inplay[inew].all(0)
        #
        self.split[inew] = True
        self.split[lower:upper] = False
        self.inplay[lower:upper] = True
        #
        self.down[inew, xnew, ynew, pnew] = arangeic(newactive_length) + lower
        #
        newimpdone = newimpossible | newdone
        inew_id = fd_index[ivector_vert[newimpdone]]
        xnew_id = xvector_vert[newimpdone]
        ynew_id = yvector_vert[newimpdone]
        pnew_id = pvector_vert[newimpdone]
        #
        self.down[inew_id, xnew_id, ynew_id, pnew_id] = -1
        #
        new_availablexyp = self.rr1.availablexyp(
            newactive_length, newactive_prod
        ).view(newactive_length, self.alpha, self.alpha, self.betaz)
        new_availablexy = new_availablexyp.any(3)
        #
        self.availablexyp[lower:upper] = new_availablexyp
        self.availablexy[lower:upper] = new_availablexy
        #
        #
        new_fdlength = self.FullData["length"]
        #
        self.FullData["info"][:, self.Pp.fulldata_location] = arangeic(
            new_fdlength
        )

    def fd_location(self, Data):
        length = Data["length"]
        assert length > 0
        return Data["info"][:, self.Pp.fulldata_location]

    def bounding_proofloop(self, Mstrat, fd_instances):
        #
        self.upperbound[fd_instances] = 1
        #
        Input = self.rr1.indexselectdata(self.FullData, fd_instances)
        #
        InitialActiveData = self.rr2.process(Input)
        activedetect, _, _ = self.rr2.filterdata(InitialActiveData)
        #
        ActivePool = self.rr1.detectsubdata(InitialActiveData, activedetect)
        #
        stepcount = 0
        for i in range(self.rr4.prooflooplength):
            stepcount += 1
            if ActivePool["length"] > 0:
                #
                print(".", end="")
                if (i % 50) == 49:
                    print(" ")
                if (i % 100) == 0:
                    print(i)
                #
                #
                ChunkData, cdetection = self.rr3.selectchunk(ActivePool)
                #
                #
                ProofCurrentData, _ = self.rr3.managesplit(
                    Mstrat, ChunkData, False
                )
                #
                #
                ActivePool = self.rr4.transitionactive(
                    ActivePool, cdetection, ProofCurrentData
                )
                if ProofCurrentData["length"] > 0:
                    current_fd_location = self.fd_location(ProofCurrentData)
                    fd_loc_list, fd_loc_counts = torch.unique(
                        current_fd_location, return_counts=True
                    )
                    self.upperbound[fd_loc_list] = (
                        self.upperbound[fd_loc_list] + fd_loc_counts
                    )
                if ActivePool["length"] == 0:
                    break
                if ActivePool["length"] > self.rr4.stopthreshold:
                    print("over threshold --------->>>>>>>>>>>>>>>>> stopping")
                    break
                #
                #
            if ActivePool["length"] == 0:
                break
            if ActivePool["length"] > self.rr4.stopthreshold:
                print("over threshold --------->>>>>>>>>>>>>>>>> stopping")
                break
            #
            #
        #
        print("|||")

    def calculate_current_upperbound(self):
        fdlength = self.FullData["length"]
        if fdlength == 0:
            print("no upper bounds to calculate")
            return
        #
        fd_split = self.split[0:fdlength]
        fd_inplay = self.inplay[0:fdlength]
        #
        fd_current = fd_inplay & ~fd_split
        #
        fd_instances = arangeic(fdlength)[fd_current]
        #
        self.bounding_proofloop(self.Mm, fd_instances)
        return

    def recursive_bound_step(self):
        #
        fdlength = self.FullData["length"]
        assert fdlength > 0
        #
        fd_up = self.up[0:fdlength]
        fd_xvalue = self.xvalue[0:fdlength]
        fd_yvalue = self.yvalue[0:fdlength]
        fd_pvalue = self.pvalue[0:fdlength]
        #
        fd_lowerbound = self.lowerbound[0:fdlength]
        fd_upperbound = self.upperbound[0:fdlength]
        fd_split = self.split[0:fdlength]
        fd_inplay = self.inplay[0:fdlength]
        fd_lookat = fd_inplay & fd_split
        fd_uppable = fd_up >= 0
        fd_availablexy = self.availablexy[0:fdlength]
        fd_lowerbound_xyp = torch.zeros(
            (fdlength, self.alpha, self.alpha, self.betaz),
            dtype=torch.int64,
            device=Dvc,
        )
        fd_upperbound_xyp = torch.zeros(
            (fdlength, self.alpha, self.alpha, self.betaz),
            dtype=torch.int64,
            device=Dvc,
        )
        #
        ivalue_uppable = fd_up[fd_uppable]
        xvalue_uppable = fd_xvalue[fd_uppable]
        yvalue_uppable = fd_yvalue[fd_uppable]
        pvalue_uppable = fd_pvalue[fd_uppable]
        #
        fd_lowerbound_xyp[
            ivalue_uppable, xvalue_uppable, yvalue_uppable, pvalue_uppable
        ] = fd_lowerbound[fd_uppable]
        fd_upperbound_xyp[
            ivalue_uppable, xvalue_uppable, yvalue_uppable, pvalue_uppable
        ] = fd_upperbound[fd_uppable]
        #
        fd_lowerbound_xy_r = (
            fd_lowerbound_xyp.sum(3).reshape(
                fdlength * self.alpha * self.alpha
            )
            + 1
        )
        fd_upperbound_xy_r = (
            fd_upperbound_xyp.sum(3).reshape(
                fdlength * self.alpha * self.alpha
            )
            + 1
        )
        fd_availablexy_r = fd_availablexy.reshape(
            fdlength * self.alpha * self.alpha
        )
        #
        fd_lowerbound_xy_r[~fd_availablexy_r] = 10000
        fd_upperbound_xy_r[~fd_availablexy_r] = 10000
        #
        fd_lowerbound_xy_rv = fd_lowerbound_xy_r.view(
            fdlength, self.alpha * self.alpha
        )
        fd_upperbound_xy_rv = fd_upperbound_xy_r.view(
            fdlength, self.alpha * self.alpha
        )
        fd_lowerbound_min, _ = torch.min(fd_lowerbound_xy_rv, 1)
        fd_upperbound_min, _ = torch.min(fd_upperbound_xy_rv, 1)
        fd_lowerbound_new = fd_lowerbound.clone()
        fd_upperbound_new = fd_upperbound.clone()
        fd_lowerbound_new[fd_lookat] = fd_lowerbound_min[fd_lookat]
        fd_upperbound_new[fd_lookat] = fd_upperbound_min[fd_lookat]
        self.lowerbound[0:fdlength] = fd_lowerbound_new
        self.upperbound[0:fdlength] = fd_upperbound_new

    def recursive_bound(self, iterations):
        #
        fdlength = self.FullData["length"]
        assert fdlength > 0
        #
        lower_all = self.lowerbound[0:fdlength].sum(0)
        upper_all = self.upperbound[0:fdlength].sum(0)
        for _ in range(iterations):
            self.recursive_bound_step()
            lower_new = self.lowerbound[0:fdlength].sum(0)
            upper_new = self.upperbound[0:fdlength].sum(0)
            if lower_new == lower_all and upper_new == upper_all:
                break
            else:
                lower_all = lower_new
                upper_all = upper_new

    def remove_from_play(self, subset):
        fdlength = self.FullData["length"]
        fd_up = self.up[0:fdlength]
        fd_up_mod = torch.clamp(fd_up, 0, fdlength)
        self.inplay[0:fdlength] = self.inplay[0:fdlength] & (~subset)
        for _ in range(100):
            inplay_count = self.inplay[0:fdlength].to(torch.int64).sum(0)
            inplay_up = self.inplay[fd_up_mod]
            self.inplay[0:fdlength] = self.inplay[0:fdlength] & inplay_up
            if self.inplay[0:fdlength].to(torch.int64).sum(0) == inplay_count:
                break

    def initial_cut(self, x, y, p):
        fdlength = self.FullData["length"]
        # assert fdlength == 28
        instance = self.down[0, x, y, p]
        subset = torch.ones((fdlength), dtype=torch.bool, device=Dvc)
        subset[0] = False
        subset[instance] = False
        self.remove_from_play(subset)

    def prune(self):
        #
        fdlength = self.FullData["length"]
        #
        fd_lowerbound = self.lowerbound[0:fdlength]
        fd_upperbound = self.upperbound[0:fdlength]
        fd_inplay = self.inplay[0:fdlength]
        attained = (fd_lowerbound == fd_upperbound) & fd_inplay
        up_mod = torch.clamp(self.up[0:fdlength], 0, fdlength)
        upperbound_up = self.upperbound[up_mod]
        nonoptimal = (fd_lowerbound + 1) >= upperbound_up
        nonoptimal[0] = False
        to_remove = nonoptimal | attained
        self.remove_from_play(to_remove)

    def combo_init(self):
        self.manage_next_stage()
        self.manage_next_stage()
        print("------ make initial cut at", self.cutx, self.cuty, self.cutp)
        self.initial_cut(self.cutx, self.cuty, self.cutp)
        self.calculate_current_upperbound()
        self.recursive_bound(100)
        self.prune()

    def combo_step(self):
        # print("------ combo segment ---------")
        checkdone = self.check_done()
        if checkdone:
            print("|||")
            return "done"
        # print("------ manage next stage")
        self.manage_next_stage()
        # print("------ calculate current upper bound")
        self.calculate_current_upperbound()
        # print("------ recursive bound")
        self.recursive_bound(100)
        # print("------ prune")
        self.prune()
        # print("------ done with combo segment ---------")
        return "continue"

    def check_done(self):
        fdlength = self.FullData["length"]
        fd_inplay = self.inplay[0:fdlength]
        inplay_count = fd_inplay.to(torch.int64).sum(0)
        if inplay_count > 1:
            return False
        else:
            assert fd_inplay[0]
            cutx = self.cutx
            cuty = self.cuty
            cutp = self.cutp
            cut_instance = self.down[0, cutx, cuty, cutp]
            assert (
                self.lowerbound[cut_instance] == self.upperbound[cut_instance]
            )
            print(
                "at initial cut location",
                cutx,
                cuty,
                cutp,
                "lower bound = upper bound =",
                itp(self.lowerbound[cut_instance]),
            )
            print("this was for sigma =", self.sigma)
            #
            lb_next = torch.zeros(
                (self.alpha, self.alpha), dtype=torch.int64, device=Dvc
            )
            ub_next = torch.zeros(
                (self.alpha, self.alpha), dtype=torch.int64, device=Dvc
            )
            for x in range(self.alpha):
                for y in range(self.alpha):
                    if self.availablexy[cut_instance, x, y]:
                        lb_next[x, y] = 1
                        ub_next[x, y] = 1
                        for p in range(self.betaz):
                            if self.availablexyp[cut_instance, x, y, p]:
                                down_xyp = self.down[cut_instance, x, y, p]
                                if down_xyp > 0:
                                    lb_next_xyp = self.lowerbound[down_xyp]
                                    assert lb_next_xyp > 0
                                    lb_next[x, y] += lb_next_xyp
                                    #
                                    ub_next_xyp = self.upperbound[down_xyp]
                                    ub_next[x, y] += ub_next_xyp
        return True

    def check_done_print(self):
        fdlength = self.FullData["length"]
        fd_inplay = self.inplay[0:fdlength]
        inplay_count = fd_inplay.to(torch.int64).sum(0)
        if inplay_count > 1:
            print("not done: there are remaining locations in play")
            return False
        else:
            assert fd_inplay[0]
            cutx = self.cutx
            cuty = self.cuty
            cutp = self.cutp
            cut_instance = self.down[0, cutx, cuty, cutp]
            assert (
                self.lowerbound[cut_instance] == self.upperbound[cut_instance]
            )
            print(
                "at initial cut location",
                cutx,
                cuty,
                cutp,
                "lower bound = upper bound =",
                itp(self.lowerbound[cut_instance]),
            )
            print("this was for sigma =", self.sigma)
            #
            lb_next = torch.zeros(
                (self.alpha, self.alpha), dtype=torch.int64, device=Dvc
            )
            ub_next = torch.zeros(
                (self.alpha, self.alpha), dtype=torch.int64, device=Dvc
            )
            for x in range(self.alpha):
                for y in range(self.alpha):
                    if self.availablexy[cut_instance, x, y]:
                        lb_next[x, y] = 1
                        ub_next[x, y] = 1
                        for p in range(self.betaz):
                            if self.availablexyp[cut_instance, x, y, p]:
                                down_xyp = self.down[cut_instance, x, y, p]
                                if down_xyp > 0:
                                    lb_next_xyp = self.lowerbound[down_xyp]
                                    assert lb_next_xyp > 0
                                    lb_next[x, y] += lb_next_xyp
                                    #
                                    ub_next_xyp = self.upperbound[down_xyp]
                                    ub_next[x, y] += ub_next_xyp
            #
            print(
                "lower bounds for the next cut locations x,y are as follows:"
            )
            print(nump(lb_next - 1))
            print("upper bounds")
            print(nump(ub_next - 1))
            print(
                "lower and upper bounds should coincide. Add 1 to plug back into the previous cut location"
            )
            print("full data length was", itp(fdlength))
        return True

    def combo_all(self):
        self.combo_init()
        for i in range(20):
            print(">>>>>>>>>>>>>>>>>>>>> combo step iteration number", i)
            step_result = self.combo_step()
            if step_result == "done":
                break
        print("combo all is completed.")
