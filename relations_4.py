import gc

import torch

from constants import Dvc
from relations_3 import Relations3
from utils import arangeic, itf, itp, itt, memReport, nump, numpr


class Relations4:
    def __init__(self, pp):
        #
        self.pp = pp
        #
        self.rr3 = Relations3(pp)
        self.rr2 = self.rr3.rr2
        self.rr1 = self.rr3.rr1
        #
        self.alpha = self.pp.alpha
        self.alpha2 = self.alpha * self.alpha
        self.alpha3 = self.alpha * self.alpha * self.alpha
        self.alpha3z = self.alpha3 + 1
        self.beta = self.pp.beta
        self.betaz = self.beta + 1
        #
        self.prooflooplength = self.pp.prooflooplength
        self.done_max = 1000  # should be self.pp.done_max
        #
        self.sleeptime = self.pp.sleeptime
        self.periodicity = self.pp.periodicity
        self.stopthreshold = self.pp.stopthreshold
        #
        self.SamplePool = self.rr1.nulldata()
        self.DroppedSamplePool = self.rr1.nulldata()
        # these are by convention active (not done or impossible)
        #
        self.donecount = itt(0)
        self.ECN = 0.0
        #
        self.proofnumber = 0
        self.allnumbers = 0
        self.proofinstance = 0
        self.dropoutratio = 1.0

    def resetsamples(self):
        self.SamplePool = self.rr1.nulldata()
        self.DroppedSamplePool = self.rr1.nulldata()
        #
        self.rr2.impossible_basic_count = 0
        self.rr2.halfones_count = 0
        self.dropoutratio = 1.0
        return

    def printexamples(self, Data):
        #
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
        ternary = Data["ternary"]
        #
        if length == 0:
            print("length 0, no examples to print")
            return
        #
        bini = ternary.to(torch.int64)
        ternary_print = bini[:, :, 0] + 2 * bini[:, :, 1] - 1
        #
        permutation = torch.randperm((length), device=Dvc)
        upper = 5
        if upper > length:
            upper = length
        for i in range(upper):
            indexi = permutation[i]
            print("---------------------------------------")
            self.rr1.printprod(Data, indexi)
            print(nump(quad[indexi]))
            print(nump(ternary_print[indexi]))
        print("---------------------------------------")

    def transitionactive(self, ActivePool, cdetection, NewActiveData):
        #
        ResidualActive = self.rr1.detectsubdata(ActivePool, ~cdetection)
        NextActivePool = self.rr1.appenddata(NewActiveData, ResidualActive)
        #
        NextActivePoolCopy = self.rr1.copydata(NextActivePool)
        self.rr1.deletedata(NextActivePool)
        #
        return NextActivePoolCopy

    def transitiondone(self, C, DonePool, DoneData, aplength):
        #
        idl = DoneData["length"]
        #
        self.donecount += idl
        #
        #
        if self.pp.verbose:
            print("new done count is", itp(self.donecount))
        #
        #
        # NewDonePool = self.rr1.nulldata()
        NewDonePool = self.rr1.appenddata(DonePool, DoneData)
        ndlength = NewDonePool["length"]
        if ndlength > self.done_max:
            # print("new done pool of length",itp(ndlength),"so we send to classifier for processing")
            print("/", end="")
            DataToProcess = self.rr1.copydata(NewDonePool)
            NewDonePool = self.rr1.nulldata()
            C.process(DataToProcess)
            #
        return NewDonePool

    def transitionsamples(self, ActivePool, DroppedPool):
        #
        # transitioning samples
        #
        slength = self.SamplePool["length"]
        aplength = ActivePool["length"]
        if aplength == 0:
            assert DroppedPool["length"] == 0
            return
        apdepth = ActivePool["depth"].to(torch.int64)
        #
        aprectangle = ActivePool["info"][
            :, self.pp.sampleinfolower : self.pp.sampleinfoupper
        ].clone()
        #
        aprange = arangeic(aplength)
        #
        # now add the new sample locations to the rectangle
        newsloc = arangeic(aplength) + slength
        #
        aprectangle[aprange, apdepth] = newsloc
        #
        # this should modify active pool outside the present function:
        ActivePool["info"][
            :, self.pp.sampleinfolower : self.pp.sampleinfoupper
        ] = aprectangle
        # that isn't needed for dropped pool since it doesn't get refered back to later
        #
        self.SamplePool = self.rr1.appenddata(self.SamplePool, ActivePool)
        self.DroppedSamplePool = self.rr1.appenddata(
            self.DroppedSamplePool, DroppedPool
        )
        #
        # print("sample pool has size",itp(self.SamplePool['length']))
        # Fws.trace("transition samples Active Pool",ActivePool,5,0,0,0)
        # Fws.trace("transition samples Sample Pool",self.SamplePool,5,0,0,0)
        # self.printsampleex()
        return

    def proofloop(self, Mstrat, Mlearn, C, Input, dropoutlimit):
        #
        self.resetsamples()
        #
        if dropoutlimit > 0:
            randomize = True
        else:
            randomize = False
        #
        InitialActiveData = self.rr2.process(Input)
        activedetect, donedetect, impossibledetect = self.rr2.filterdata(
            InitialActiveData
        )
        implength = impossibledetect.to(torch.int).sum(0)
        donelength = donedetect.to(torch.int).sum(0)
        if self.pp.verbose:
            print(
                "initial filter finds",
                itp(implength),
                "impossibilities and",
                itp(donelength),
                "instances already done",
            )
        ActivePool = self.rr1.detectsubdata(InitialActiveData, activedetect)
        #
        napcount = 0
        #
        DonePool = self.rr1.detectsubdata(InitialActiveData, donedetect)
        self.donecount = itt(0)
        if ActivePool["length"] == 0:
            DonePool = self.transitiondone(
                DonePool, self.rr1.nulldata(), ActivePool["length"]
            )
        #
        #
        self.ECN = itt(ActivePool["length"]).to(torch.float).clone()
        EDN = self.ECN.clone()
        if self.pp.verbose:
            print(
                "starting with ECN = EDN from initial active pool",
                numpr(self.ECN, 1),
            )
        #
        if dropoutlimit > 0:
            ActivePool, DroppedPool, newsum, droppedsum = self.dropoutdata(
                Mlearn, ActivePool, dropoutlimit
            )
            if self.pp.dropout_style == "adaptive":
                activelengthf = (
                    itt(ActivePool["length"]).clone().to(torch.float)
                )
                self.ECN = activelengthf + droppedsum
                EDN = 0.0
        #
        stepcount = 0
        for i in range(self.prooflooplength):
            stepcount += 1
            prooflength = i
            if ActivePool["length"] > 0:
                #
                PreAPL = itt(ActivePool["length"]).clone()
                #
                if self.pp.verbose:
                    print("= = = = = =  loop", i, "= = = = =", end=" ")
                    print(
                        itp(self.proofnumber),
                        "/",
                        itp(self.allnumbers),
                        "<",
                        itp(self.proofinstance),
                        "> = = = =",
                    )
                else:
                    print(".", end="")
                    if (i % 50) == 49:
                        print(" ")
                    if (i % 100) == 0:
                        print(i)
                napcount += 1
                #
                ChunkData, cdetection = self.rr3.selectchunk(ActivePool)
                #
                #
                CurrentData, DoneData = self.rr3.managesplit(
                    Mstrat, ChunkData, randomize
                )
                #
                #
                ActivePool = self.transitionactive(
                    ActivePool, cdetection, CurrentData
                )
                # do the following before dropout
                if dropoutlimit == 0:
                    EDN = itt(ActivePool["length"]).clone().to(torch.float)
                    self.ECN += (
                        itt(CurrentData["length"]).clone().to(torch.float)
                    )
                    if self.ECN > HST.proof_nodes_max:
                        print("break after maximum proof nodes")
                        break
                #
                if dropoutlimit > 0:
                    if (
                        self.pp.dropout_style == "regular"
                        or self.pp.dropout_style == "uniform"
                    ):
                        PostAPL = (
                            itt(ActivePool["length"]).clone().to(torch.float)
                        )
                        ratio = PostAPL / PreAPL
                        EDN *= ratio
                        self.ECN += EDN
                    (
                        ActivePool,
                        DroppedPool,
                        newsum,
                        droppedsum,
                    ) = self.dropoutdata(Mlearn, ActivePool, dropoutlimit)
                    #
                    self.transitionsamples(ActivePool, DroppedPool)
                    #
                    if self.pp.dropout_style == "adaptive":
                        activelengthf = (
                            itt(ActivePool["length"]).clone().to(torch.float)
                        )
                        self.ECN += activelengthf + droppedsum
                        EDN = 0.0
                    #
                    #
                    #
                    if self.pp.verbose:
                        self.printexamples(ActivePool)
                #
                #
                if self.pp.verbose:
                    print(
                        "Active Pool has length",
                        itp(ActivePool["length"]),
                        end=" ",
                    )
                    #
                    print(
                        "treated Chunk Data of length",
                        itp(ChunkData["length"]),
                    )
                    print(
                        "yielding Current Data of length",
                        itp(CurrentData["length"]),
                    )
                    print(
                        "net active data gained",
                        itp(CurrentData["length"] - ChunkData["length"]),
                    )
                    #
                    print(
                        "Active Pool has length",
                        itp(ActivePool["length"]),
                        end=" ",
                    )
                DonePool = self.transitiondone(
                    C, DonePool, DoneData, ActivePool["length"]
                )
                #
                gcc = gc.collect()
                if self.pp.verbose:
                    memReport("mg")
                    #
                    #
                    if 0 < dropoutlimit <= self.pp.chunksize:
                        print(
                            "Estimated nodes at this depth",
                            numpr(EDN, 1),
                            "estimated cumulative nodes",
                            numpr(self.ECN, 1),
                        )
                    if dropoutlimit == 0:
                        print(
                            "current nodes",
                            numpr(EDN, 1),
                            "cumulative nodes",
                            numpr(self.ECN, 1),
                        )
                #
                if ActivePool["length"] == 0:
                    break
                if ActivePool["length"] > self.stopthreshold:
                    print("over threshold --------->>>>>>>>>>>>>>>>> stopping")
                    break
                #
                #
            if ActivePool["length"] == 0:
                break
            if ActivePool["length"] > self.stopthreshold:
                print("over threshold --------->>>>>>>>>>>>>>>>> stopping")
                break
            #
            # if (napcount%self.periodicity) == 0:
            # siesta(self.sleeptime)
            #
        #
        print("|||")
        activelength = ActivePool["length"]
        donelength = DonePool["length"]
        if donelength > 0:
            C.process(DonePool)
            DonePool = self.rr1.nulldata()
        #
        if dropoutlimit == 0:
            cumulative_nodes = torch.round(self.ECN).to(torch.int64)
            HST.record_full_proof(
                Mstrat, stepcount, cumulative_nodes, self.donecount
            )
        else:
            HST.record_dropout_proof(
                self.pp.dropout_style, dropoutlimit, stepcount, self.ECN
            )
        #
        if self.pp.verbose:
            if activelength > 0:
                print(
                    "there remained",
                    itp(activelength),
                    "active locations",
                    end=" ",
                )
            else:
                print("no further active locations", end=" ")
            print("done pool has length", itp(donelength))
            print(
                "Estimated Cumulative Nodes at end of proof",
                numpr(self.ECN, 1),
            )
            print("done count is", itp(self.donecount))
            print(
                "impossible basic count is",
                itp(self.rr2.impossible_basic_count),
            )
            print("half ones count is", itp(self.rr2.halfones_count))
        return True, ActivePool, DonePool, prooflength

    def dropoutdata(self, M, Data, dropoutlimit):
        if self.pp.dropout_style == "regular":
            NewData, DroppedData = self.dropoutdataRegular(Data, dropoutlimit)
            newsum = 0.0
            droppedsum = 0.0
        if self.pp.dropout_style == "adaptive":
            (
                NewData,
                DroppedData,
                newsum,
                droppedsum,
            ) = self.dropoutdataAdaptive(M, Data, dropoutlimit)
        if self.pp.dropout_style == "uniform":
            NewData, DroppedData, newsum, droppedsum = self.dropoutdataUniform(
                M, Data, dropoutlimit
            )
        #
        return NewData, DroppedData, newsum, droppedsum

    def dropoutdataRegular(self, Data, dropoutlimit):
        length = Data["length"]
        if length == 0:
            return Data, self.rr1.nulldata()
        permutation = torch.randperm(length, device=Dvc)
        upper = dropoutlimit
        if upper > length:
            NewData = self.rr1.copydata(Data)
            DroppedData = self.rr1.nulldata()
        else:
            indices = permutation[0:upper]
            indices_dropped = permutation[upper:length]
            NewData = self.rr1.indexselectdata(Data, indices)
            DroppedData = self.rr1.indexselectdata(Data, indices_dropped)
        return NewData, DroppedData

    def extent_sliced(self, M, Data):
        #
        length = Data["length"]
        if length <= 1000:
            extent_log = M.network(Data).detach()
            extent_log = torch.clamp(extent_log, 0.0, 8.0)
            extent = 10 ** extent_log
            return extent
        extent = torch.zeros((length), dtype=torch.float, device=Dvc)
        lrange = arangeic(length)
        #
        lower = 0
        for i in range(length):
            upper = lower + 1000
            if upper > length:
                upper = length
            indices = lrange[lower:upper]
            DataSlice = self.rr1.indexselectdata(Data, indices)
            extent_log = M.network(DataSlice).detach()
            extent_log = torch.clamp(extent_log, 0.0, 8.0)
            extent[lower:upper] = 10 ** extent_log
            lower = upper
            if upper >= length:
                break
        return extent

    def dropoutdataAdaptive(self, M, Data, dropoutlimit):
        length = Data["length"]
        if length == 0:
            return Data, self.rr1.nulldata(), 0.0, 0.0
        #
        extent = self.extent_sliced(M, Data)
        denom = extent.sum(0)
        proba = (dropoutlimit * extent) / denom
        tirage = torch.rand(length, device=Dvc)
        detection = tirage < proba
        indices1 = arangeic(length)[detection]
        dlength = detection.to(torch.int64).sum(0)
        if dlength > dropoutlimit:
            indices1 = arangeic(length)[detection]
            permutation = torch.randperm(dlength)
            indices2 = permutation[dropoutlimit:dlength]
            indices3 = indices1[indices2]
            detection[indices3] = False
        if length <= dropoutlimit:
            NewData = self.rr1.copydata(Data)
            DroppedData = self.rr1.nulldata()
            newsum = extent.sum(0)
            droppedsum = 0.0
        else:
            NewData = self.rr1.detectsubdata(Data, detection)
            DroppedData = self.rr1.detectsubdata(Data, (~detection))
            newsum = (extent[detection]).sum(0)
            droppedsum = (extent[~detection]).sum(0)
        return NewData, DroppedData, newsum, droppedsum

    def dropoutdataUniform(self, M, Data, dropoutlimit):
        length = Data["length"]
        if length == 0:
            return Data, self.rr1.nulldata(), 0.0, 0.0
        #
        extent = self.extent_sliced(M, Data)
        #
        values, sort_indices = torch.sort(extent, 0)
        fraction = itf(length) / itf(dropoutlimit)
        epsilon_multiplier = itf(length - dropoutlimit) / itf(length)
        epsilon_multiplier = torch.clamp(epsilon_multiplier, 0.0, 1.0)
        drange = arangeic(dropoutlimit).to(torch.float)
        tirage = torch.rand(dropoutlimit, device=Dvc)
        tirage2 = torch.rand(dropoutlimit, device=Dvc) - 0.5
        tirage2vx = tirage2.view(dropoutlimit, 1).expand(
            dropoutlimit, dropoutlimit
        )
        irange = (
            arangeic(dropoutlimit)
            .view(dropoutlimit, 1)
            .expand(dropoutlimit, dropoutlimit)
        )
        jrange = (
            arangeic(dropoutlimit)
            .view(1, dropoutlimit)
            .expand(dropoutlimit, dropoutlimit)
        )
        tirage2_triangle = tirage2vx * ((irange > jrange).to(torch.float))
        tirage2_integral = tirage2_triangle.sum(0)
        epsilon = tirage * epsilon_multiplier
        drange_mod = drange + epsilon + (0.05 * tirage2_integral)
        #
        float_indices = drange_mod * fraction
        round_indices = torch.round(float_indices).to(torch.int64)
        round_indices = torch.clamp(round_indices, 0, length - 1)
        combined_indices = sort_indices[round_indices]
        detection = torch.zeros((length), dtype=torch.bool, device=Dvc)
        detection[combined_indices] = True
        #
        if length <= dropoutlimit:
            NewData = self.rr1.copydata(Data)
            DroppedData = self.rr1.nulldata()
            newsum = extent.sum(0)
            droppedsum = 0.0
        else:
            NewData = self.rr1.detectsubdata(Data, detection)
            DroppedData = self.rr1.detectsubdata(Data, (~detection))
            newsum = (extent[detection]).sum(0)
            droppedsum = (extent[~detection]).sum(0)
        return NewData, DroppedData, newsum, droppedsum

    def printsampleex(self):
        samplelength = self.SamplePool["length"]
        if samplelength == 0:
            return
        upper = 20
        if upper > samplelength:
            upper = samplelength
        permutation = torch.randperm(samplelength, device=Dvc)
        depth = self.SamplePool["depth"]
        points = self.SamplePool["info"][:, self.pp.samplepoints]
        for i in range(upper):
            ip = permutation[i]
            print(
                "number",
                ip,
                "depth",
                itp(depth[ip]),
                "points",
                itp(points[ip]),
            )
        return
