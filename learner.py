import torch
from matplotlib import pyplot as plt

from constants import Dvc
from historical import Historical
from utils import CoherenceError, arangeic, itf, itp, nump, numpr


class Learner:  # training the neural networks
    def __init__(self, Rr4, HST: Historical):
        #
        #
        self.HST = HST
        self.rr4 = Rr4
        self.pp = self.rr4.pp
        self.rr3 = self.rr4.rr3
        self.rr2 = self.rr4.rr2
        self.rr1 = self.rr4.rr1
        self.alpha = self.pp.alpha
        self.alpha2 = self.pp.alpha2
        self.alpha3 = self.pp.alpha3
        self.alpha3z = self.pp.alpha3z
        self.beta = self.pp.beta
        self.betaz = self.pp.betaz
        #
        # self.mm = Mm
        self.explore_max = self.pp.explore_max
        self.examples_max = self.pp.examples_max
        self.new_examples_max = self.pp.new_examples_max
        self.new_explore_max = self.pp.new_explore_max
        self.outlier_max = self.pp.outlier_max
        self.new_outliers_max = self.pp.new_outliers_max
        #
        self.trainingprint = True
        #
        # all the examples are by convention active (not done or impossible)
        self.OutlierPrePool = self.rr1.nulldata()
        self.ExplorePrePool = self.rr1.nulldata()
        self.ExamplesPrePool = self.rr1.nulldata()
        self.Examples = self.rr1.nulldata()
        self.extent_log = None  # log10 of the number of nodes below that node, including that one, note everything is active
        # so this number of nodes is never 0
        self.localscores = None  # at (x,y) it is log10 of the sum score over (x,y,p), plus 1 for the upper node itself
        #
        # self.sga = SymmetricGroup(self.alpha)
        # self.sgb = SymmetricGroup(self.beta)
        #
        self.globalL1level = 1.0

    def pruneExamples(
        self, M
    ):  # remove the last ones, those were the earliest added
        #
        elength = self.Examples["length"]
        epplength = self.ExamplesPrePool["length"]
        xlength = self.ExplorePrePool["length"]
        olength = self.OutlierPrePool["length"]
        #
        if elength > self.examples_max:
            # self.specialPruneExamples(M)
            permutation = torch.randperm(elength, device=Dvc)
            indices = permutation[0 : self.examples_max]
            self.Examples = self.rr1.indexselectdata(self.Examples, indices)
            self.extent_log = self.extent_log[indices]
            self.localscores = self.localscores[indices]
        if epplength > self.examples_max:
            permutation = torch.randperm(epplength, device=Dvc)
            indices = permutation[0 : self.examples_max]
            self.ExamplesPrePool = self.rr1.indexselectdata(
                self.ExamplesPrePool, indices
            )
        if xlength > self.explore_max:
            permutation = torch.randperm(xlength, device=Dvc)
            indices = permutation[0 : self.explore_max]
            self.ExplorePrePool = self.rr1.indexselectdata(
                self.ExplorePrePool, indices
            )
        if olength > self.outlier_max:
            permutation = torch.randperm(olength, device=Dvc)
            indices = permutation[0 : self.outlier_max]
            self.OutlierPrePool = self.rr1.indexselectdata(
                self.OutlierPrePool, indices
            )
        return

    def check_availability(self, Data, comment):
        #
        prod = Data["prod"]
        #
        available = (((prod.any(3)).all(2)).all(1)).all(0)
        if not available:
            print("with comment", comment)
            raise CoherenceError(
                "availability problem in data that should be active"
            )
        return

    def prepoolSamples(
        self, M, Data, new_examples
    ):  # prepend new ones so that the earliest ones were the last ones
        #
        #
        splength = Data["length"]
        #
        if splength == 0:
            return
        permutation = torch.randperm(splength, device=Dvc)
        upper = new_examples
        if upper > splength:
            upper = splength
        indices = permutation[0:upper]
        NewExamples = self.rr1.indexselectdata(Data, indices)
        self.check_availability(NewExamples, "prepoolSamples")
        self.ExamplesPrePool = self.rr1.appenddata(
            NewExamples, self.ExamplesPrePool
        )
        #
        self.pruneExamples(M)
        return

    def prepoolExplore(
        self, M, Data, new_examples
    ):  # prepend new ones so that the earliest ones were the last ones
        #
        #
        splength = Data["length"]
        #
        if splength == 0:
            return
        ppe_length = torch.round(itf(new_examples) / M.average_local_loss).to(
            torch.int64
        )
        rectangle = (
            arangeic(splength)
            .view(splength, 1)
            .expand(splength, ppe_length)
            .reshape(splength * ppe_length)
        )
        permutation = torch.randperm(splength * ppe_length, device=Dvc)
        upper = ppe_length
        #
        indices = rectangle[permutation[0:upper]]
        NewExamples = self.rr1.indexselectdata(Data, indices)
        self.check_availability(NewExamples, "prepoolExplore")
        self.ExplorePrePool = self.rr1.appenddata(
            NewExamples, self.ExplorePrePool
        )
        #
        self.pruneExamples(M)
        return

    def transferexamples(
        self, M, TransferData, extent, localscore, epp_throw_detection
    ):
        # detection tells us the ones to remove from the epp
        if self.Examples["length"] == 0:
            self.extent_log = extent
            self.localscores = localscore
        else:
            self.extent_log = torch.cat((extent, self.extent_log), 0)
            self.localscores = torch.cat((localscore, self.localscores), 0)
        self.Examples = self.rr1.appenddata(TransferData, self.Examples)
        #
        keep_detection = ~epp_throw_detection
        self.ExamplesPrePool = self.rr1.detectsubdata(
            self.ExamplesPrePool, keep_detection
        )
        #
        delta = self.abs_diff_sup(M, TransferData, localscore)
        seuil = self.pp.outlier_threshold * M.average_local_loss
        NewOutliers = self.rr1.detectsubdata(TransferData, (delta > seuil))
        self.OutlierPrePool = self.rr1.appenddata(
            NewOutliers, self.OutlierPrePool
        )
        #
        self.pruneExamples(M)
        return

    def scoreExamples(self, M):
        epplength = self.ExamplesPrePool["length"]
        if epplength == 0:
            print("no examples to locally score")
            return
        #
        permutation = torch.randperm(epplength, device=Dvc)
        upper = self.new_examples_max
        if upper > epplength:
            upper = epplength
        indices = permutation[0:upper]
        #
        epp_throw_detection = torch.zeros(
            (epplength), dtype=torch.bool, device=Dvc
        )
        epp_throw_detection[indices] = True
        #
        TransferData = self.rr1.indexselectdata(self.ExamplesPrePool, indices)
        #
        print("transfering data of length", itp(TransferData["length"]))
        #
        xyscore_log, xyscore_min, LocalExamples = self.calculatescoresLocal(
            M, TransferData
        )
        #
        self.transferexamples(
            M, TransferData, xyscore_min, xyscore_log, epp_throw_detection
        )
        #
        self.check_availability(
            LocalExamples, "LocalExamples in scoreExamples"
        )
        self.prepoolExplore(M, LocalExamples, self.new_examples_max)
        #
        print("examples has size", itp(self.Examples["length"]))
        print("example pre pool has size", itp(self.ExamplesPrePool["length"]))
        return

    def scoreExplore(self, M):
        xlength = self.ExplorePrePool["length"]
        epplength = self.ExamplesPrePool["length"]
        if xlength == 0:
            print("no examples to locally score")
            return
        #
        permutation = torch.randperm(xlength, device=Dvc)
        current_explore = 2 * self.new_examples_max
        if current_explore > self.new_explore_max:
            current_explore = self.new_explore_max
        upper = current_explore
        if upper > xlength:
            upper = xlength
        indices = permutation[0:upper]
        #
        epp_throw_detection = torch.zeros(
            (epplength), dtype=torch.bool, device=Dvc
        )
        #
        TransferData = self.rr1.indexselectdata(self.ExplorePrePool, indices)
        #
        print(
            "transfering explore data of length", itp(TransferData["length"])
        )
        #
        xyscore_log, xyscore_min, LocalExamples = self.calculatescoresLocal(
            M, TransferData
        )
        #
        self.transferexamples(
            M, TransferData, xyscore_min, xyscore_log, epp_throw_detection
        )
        #
        self.check_availability(LocalExamples, "LocalExamples in scoreExplore")
        self.prepoolExplore(M, LocalExamples, current_explore)
        #
        print("examples has size", itp(self.Examples["length"]))
        print("explore pre pool has size", itp(self.ExplorePrePool["length"]))
        return

    def scoreOutlier(self, M):
        xlength = self.OutlierPrePool["length"]
        epplength = self.ExamplesPrePool["length"]
        if xlength == 0:
            print("no outliers to locally score")
            return
        #
        permutation = torch.randperm(xlength, device=Dvc)
        #
        upper = self.new_outliers_max
        if upper > xlength:
            upper = xlength
        indices = permutation[0:upper]
        #
        epp_throw_detection = torch.zeros(
            (epplength), dtype=torch.bool, device=Dvc
        )
        #
        TransferData = self.rr1.indexselectdata(self.OutlierPrePool, indices)
        #
        print(
            "transfering outlier data of length", itp(TransferData["length"])
        )
        #
        xyscore_log, xyscore_min, LocalExamples = self.calculatescoresLocal(
            M, TransferData
        )
        #
        delta = self.abs_diff_sup(M, TransferData, xyscore_log)
        seuil = self.pp.outlier_threshold * M.average_local_loss
        indices_throw = indices[(delta < seuil)]
        throw_detection = torch.zeros((xlength), dtype=torch.bool, device=Dvc)
        throw_detection[indices_throw] = True
        self.OutlierPrePool = self.rr1.detectsubdata(
            self.OutlierPrePool, (~throw_detection)
        )
        #
        self.transferexamples(
            M, TransferData, xyscore_min, xyscore_log, epp_throw_detection
        )
        #
        print("outlier pre pool has size", itp(self.OutlierPrePool["length"]))
        return

    def abs_diff_sup(self, M, Data, xyscore_log):
        #
        a = self.alpha
        a2 = self.alpha2
        #
        length = Data["length"]
        prod = Data["prod"]
        #
        availablexyf = self.rr1.availablexy(length, prod).view(length, a, a)
        net2 = M.network2(Data)
        predicted = net2.view(length, a, a).detach()
        delta_abs = torch.abs((availablexyf * (xyscore_log - predicted))).view(
            length, a2
        )
        delta_sup, indices = torch.max(delta_abs, 1)
        return delta_sup

    def addscoredexamples(self, M):
        SamplePool = self.rr4.SamplePool
        DroppedPool = self.rr4.DroppedSamplePool
        #
        splength = SamplePool["length"]
        dplength = DroppedPool["length"]
        #
        sprectangle = SamplePool["info"][
            :, self.pp.sampleinfolower : self.pp.sampleinfoupper
        ].clone()
        #
        splrangevxr = (
            arangeic(splength)
            .view(splength, 1)
            .expand(splength, 200)
            .reshape(splength * 200)
        )
        sprectangler = sprectangle.reshape(splength * 200)
        sp_detection = (sprectangler >= 0) & (sprectangler < splrangevxr)
        #
        ivector = splrangevxr[sp_detection]
        jvector = sprectangler[sp_detection]
        # note that jvector < ivector because of the second condition in sp_detection
        # jvector is any previous location strictly above the ivector location in the proof tree
        #
        incidence = torch.zeros(
            (splength, splength), dtype=torch.float, device=Dvc
        )
        #
        incidence[ivector, jvector] = 1.0
        #
        #
        spnodes = (
            incidence.sum(0) + 1.0
        )  # this should be the number of nodes below that location, including that location
        #
        if dplength > 0:
            dprectangle = DroppedPool["info"][
                :, self.pp.sampleinfolower : self.pp.sampleinfoupper
            ].clone()
            #
            dincidence = torch.zeros(
                (dplength, splength), dtype=torch.float, device=Dvc
            )
            #
            dplrangevxr = (
                arangeic(dplength)
                .view(dplength, 1)
                .expand(dplength, 200)
                .reshape(dplength * 200)
            )
            dprectangler = dprectangle.reshape(dplength * 200)
            dp_detection = dprectangler >= 0
            #
            idvector = dplrangevxr[dp_detection]
            jdvector = dprectangler[dp_detection]
            #
            dincidence[idvector, jdvector] = 1.0
            #
            #
            dpextent_log = M.network(DroppedPool).detach()
            # this approximates log10 of (the number of nodes at or below a dropped location)
            dpextent_log = torch.clamp(dpextent_log, 0.0, 9.0)
            dpextent = 10 ** dpextent_log
            dpextentv = dpextent.view(1, dplength)
            extent_transfer = (torch.matmul(dpextentv, dincidence)).view(
                splength
            )
            #
            extent = spnodes + extent_transfer
        else:
            extent = spnodes
        #
        logextent = torch.log10(extent)
        #
        #
        permutation = torch.randperm(splength, device=Dvc)
        upper = self.new_examples_max
        if upper > splength:
            upper = splength
        indices = permutation[0:upper]
        TransferData = self.rr1.indexselectdata(SamplePool, indices)
        transfer_extent = logextent[indices]
        #
        epp_throw_detection = torch.zeros(
            (self.ExamplesPrePool["length"]), dtype=torch.bool, device=Dvc
        )
        #
        xyscore_log, xyscore_min, LocalExamples = self.calculatescoresLocal(
            M, TransferData
        )
        #
        positivephase = TransferData["info"][:, self.pp.phase] > 0
        phased_extent = transfer_extent.clone()
        phased_extent[positivephase] = xyscore_min[positivephase]
        #
        self.transferexamples(
            M, TransferData, phased_extent, xyscore_log, epp_throw_detection
        )
        #
        return

    def noisetensor(self, thetensor):
        #
        length = len(thetensor)
        tirage = torch.rand(length, device=Dvc)
        noiselevel = self.HST.noiselevel(
            self.pp, torch.tensor(self.HST.training_counter, device=Dvc)
        ).item()
        bruit = tirage < noiselevel
        ntensor = ((~bruit) & thetensor) | (bruit & (~thetensor))
        return ntensor

    def noise(self, Data):
        #
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        length = Data["length"]
        prodv = Data["prod"].view(length * a * a * bz)
        leftv = Data["left"].view(length * a * bz * 2)
        rightv = Data["right"].view(length * bz * a * 2)
        ternaryv = Data["ternary"].view(length * a * a * a * 2)
        #
        nprod = self.noisetensor(prodv).view(length, a, a, bz)
        nleft = self.noisetensor(leftv).view(length, a, bz, 2)
        nright = self.noisetensor(rightv).view(length, bz, a, 2)
        nternary = self.noisetensor(ternaryv).view(length, a, a, a, 2)
        #
        NoiseData = self.rr1.copydata(Data)
        NoiseData["prod"] = nprod
        NoiseData["left"] = nleft
        NoiseData["right"] = nright
        NoiseData["ternary"] = nternary
        #
        return NoiseData

    def printexamplescores(self, number):
        ExamplePool = self.Examples
        xplength = ExamplePool["length"]
        xpdepth = ExamplePool["depth"]
        if xplength == 0:
            print("no examples to print")
            return
        #
        print("example pool has", itp(xplength), "elements")
        #
        permutation = torch.randperm(xplength, device=Dvc)
        #
        upper = number
        if upper > xplength:
            upper = xplength
        for i in range(upper):
            ip = permutation[i]
            idepth = xpdepth[ip]
            iextent = self.extent_log[ip]
            print(
                "sample number",
                itp(ip),
                "depth",
                itp(idepth),
                "log extent",
                numpr(iextent, 2),
            )
        return

    def selectminibatch(self, minibatchsize):
        ExamplePool = self.Examples
        xplength = ExamplePool["length"]
        score = self.extent_log
        if xplength < 10:
            print("not enough examples to train on")
            return False, None, None
        #
        permutation = torch.randperm(xplength, device=Dvc)
        upper = minibatchsize
        if upper > xplength:
            upper = xplength
        indices = permutation[0:upper]
        DataBatch = self.rr1.indexselectdata(ExamplePool, indices)
        scorebatch = score[indices]
        return True, DataBatch, scorebatch

    def trainingGlobal(
        self,
        M,
        numberofbatches,
        iterationsperbatch,
        style,
        minibatchsize,
        partname,
    ):
        #
        if style != "score-A" and style != "score-B" and style != "score-C":
            raise CoherenceError(
                "only allowed styles are score-A or score-B or score-C"
            )
        #
        if self.trainingprint:
            print(
                "/",
                style,
                numberofbatches,
                iterationsperbatch,
                partname,
                "/",
                end=" ",
            )
        for s in range(numberofbatches):
            smb, DataBatch, scorebatch = self.selectminibatch(minibatchsize)
            if not smb:
                print("exit training")
                return
            #
            for i in range(iterationsperbatch):
                #
                M.optimizer.zero_grad()
                #
                NoiseData = self.noise(DataBatch)
                #
                predictedscore = M.network(NoiseData)
                #
                if style == "score-A":
                    loss = M.criterionA(predictedscore, scorebatch)
                if style == "score-B":
                    loss = M.criterionB(predictedscore, scorebatch)
                if style == "score-C":
                    lossA = M.criterionA(predictedscore, scorebatch)
                    lossB = M.criterionB(predictedscore, scorebatch)
                    loss = (lossA + lossB) / 2
                loss.backward()
                M.optimizer.step()
            #
        #
        print("-", end=" ")
        #
        return

    def printlossaftertrainingGlobal(self, M, minibatchsize, topicture):
        #
        smb, DataBatch, scorebatch = self.selectminibatch(minibatchsize)
        if not smb:
            print("data too small")
            return
        #
        mblength = DataBatch["length"]
        predictedscore = M.network(DataBatch)
        lossa = M.criterionA(predictedscore, scorebatch)
        lra = numpr(lossa, 3)
        lossb = M.criterionB(predictedscore, scorebatch)
        lrb = numpr(lossb, 3)
        #
        with torch.no_grad():
            self.globalL1level += lossa
            self.globalL1level *= 0.5
            self.globalL1level = torch.clamp(self.globalL1level, 0.005, 1.0)
        #
        print(
            "on ",
            itp(mblength),
            "values network -- L1 loss",
            lra,
            "MSE loss",
            lrb,
        )
        #
        if topicture:
            #
            print("global L1 loss level", numpr(self.globalL1level, 4))
            #
            self.HST.record_loss("global", lossa, lossb)
            #
            dotsize = torch.zeros((mblength), dtype=torch.int, device=Dvc)
            dotsize[:] = 2
            dotsize_np = nump(dotsize)
            #
            calcscore_npr = numpr(scorebatch, 3)
            predscore_npr = numpr(predictedscore, 3)
            #
            scoremax, index = torch.max(scorebatch, 0)
            linelimit = numpr(scoremax, 1)
            #
            #
            plt.clf()
            plt.scatter(calcscore_npr, predscore_npr, dotsize_np)
            #
            plt.plot([0.0, linelimit], [0.0, 0.0], "g-", lw=1)
            plt.plot([0.0, 0.0], [0.0, linelimit], "g-", lw=1)
            plt.plot([0.0, linelimit], [0.0, linelimit], "r-", lw=1)
            plt.show()
        return

    def learningGlobal(self, M, globaliterations):
        #
        self.printlossaftertrainingGlobal(M, 500, True)
        #
        tweak_cursor = self.HST.global_tweak_cursor
        tdensity = self.pp.tweak_density * (
            self.pp.tweak_decay ** tweak_cursor
        )
        tepsilon = self.pp.tweak_epsilon * (
            self.pp.tweak_decay ** tweak_cursor
        )
        self.HST.global_tweak_cursor += 1
        print(
            "tweaking global network at cursor",
            itp(tweak_cursor),
            "with density",
            numpr(tdensity, 4),
            "and epsilon",
            numpr(tepsilon, 4),
        )
        M.tweak_network(M.network, tdensity, tepsilon)
        self.printlossaftertrainingGlobal(M, 500, True)
        print("training", end=" ")
        #
        explore_pre_length = self.ExplorePrePool["length"]
        example_pre_length = self.ExamplesPrePool["length"]
        example_length = self.Examples["length"]
        self.HST.record_training(
            "global",
            globaliterations,
            explore_pre_length,
            example_pre_length,
            example_length,
        )
        #
        for g in range(globaliterations):
            print("/", end=" ")
            self.trainingGlobal(M, 2, 20, "score-C", 20, "mb20")
            self.trainingGlobal(M, 1, 30, "score-C", 30, "mb30")
            self.trainingGlobal(M, 2, 10, "score-C", 40, "mb40")
            self.trainingGlobal(M, 5, 10, "score-C", 60, "mb60")
            self.trainingGlobal(M, 3, 8, "score-C", 20, "mb20")
            self.trainingGlobal(M, 3, 5, "score-C", 40, "mb40")
            self.trainingGlobal(M, 5, 3, "score-C", 30, "mb30")
            #
            print(" ")
            self.printlossaftertrainingGlobal(M, 300, False)
            print("  ")
        self.printlossaftertrainingGlobal(M, 500, True)
        print("=================    end score training   =================")
        return

    ######## Local stuff

    def calculatescoresLocal(self, M, Data):
        #
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        length = Data["length"]
        prod = Data["prod"]
        #
        xypscore_exp = torch.zeros(
            (length, a, a, bz), dtype=torch.float, device=Dvc
        )
        #
        availablexyp = self.rr1.availablexyp(length, prod)
        availablexypr = availablexyp.reshape(length * a * a * bz)
        #
        lrangevxr = (
            arangeic(length)
            .view(length, 1, 1, 1)
            .expand(length, a, a, bz)
            .reshape(length * a * a * bz)
        )
        xrangevxr = (
            arangeic(a)
            .view(1, a, 1, 1)
            .expand(length, a, a, bz)
            .reshape(length * a * a * bz)
        )
        yrangevxr = (
            arangeic(a)
            .view(1, 1, a, 1)
            .expand(length, a, a, bz)
            .reshape(length * a * a * bz)
        )
        prangevxr = (
            arangeic(bz)
            .view(1, 1, 1, bz)
            .expand(length, a, a, bz)
            .reshape(length * a * a * bz)
        )
        #
        ivector = lrangevxr[availablexypr]
        xvector = xrangevxr[availablexypr]
        yvector = yrangevxr[availablexypr]
        pvector = prangevxr[availablexypr]
        #
        NewData = self.rr1.upsplitting(
            Data, ivector, xvector, yvector, pvector
        )
        #
        #
        ndlength = NewData["length"]
        #
        #
        LocalExamples = self.rr1.nulldata()
        detection = torch.zeros((ndlength), dtype=torch.bool, device=Dvc)
        newextent_exp = torch.zeros((ndlength), dtype=torch.float, device=Dvc)
        # that should be the (approximation of) the number of nodes below and including that node resulting from (i,x,y,p)
        newactive = torch.zeros((ndlength), dtype=torch.bool, device=Dvc)
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
            newactive_s, newdone_s, newimpossible_s = self.rr2.filterdata(
                AssocNewDataSlice
            )
            #
            ActiveNewDataSlice = self.rr1.detectsubdata(
                AssocNewDataSlice, newactive_s
            )
            LocalExamples = self.rr1.appenddata(
                ActiveNewDataSlice, LocalExamples
            )
            #
            predictedscore_s = M.network(AssocNewDataSlice).detach()
            if torch.isnan(predictedscore_s).any(0):
                raise CoherenceError("predicted score nan")
            # recall that approximates log10 of (the number of nodes below and including that node)
            predictedscore_s_clamp = torch.clamp(predictedscore_s, 0.0, 8.0)
            predictedscore_s_exp = 10.0 ** predictedscore_s_clamp
            predictedscore_s_exp[newdone_s] = 0.0
            predictedscore_s_exp[newimpossible_s] = 0.1
            newextent_exp[lower:upper] = predictedscore_s_exp
            newactive[lower:upper] = newactive_s
            lower = upper
            if lower >= ndlength:
                break
        #
        xypscore_exp[ivector, xvector, yvector, pvector] = newextent_exp
        #
        xyscore_sum = xypscore_exp.sum(3)
        #
        xyscore_log = torch.log10(
            xyscore_sum + 1.0
        )  # here the +1. is for the upper node itself.
        #
        # now replace the global scores by these ones too: here output the min of availables
        availablexyr = self.rr1.availablexy(length, prod).view(length * a2)
        xyscorer = xyscore_log.view(length * a2)
        xyscorer_mod = torch.clamp(xyscorer, 0.0, 8.0)
        xyscorer_mod[~availablexyr] = 20.0
        xyscore_min, xysm_indices = torch.min(xyscorer_mod.view(length, a2), 1)
        assert (xyscore_min < 10.0).all(0)
        #
        return xyscore_log, xyscore_min, LocalExamples

    def printsomelocalscores(self, howmany):
        elength = self.Examples["length"]
        if elength == 0:
            print("no examples")
            return
        lsk = self.localscores_known
        detection = lsk
        detection_length = detection.to(torch.int64).sum(0)
        if detection_length == 0:
            print("no known score locations to print")
            return
        ivector = arangeic(elength)[detection]
        permutation = torch.randperm(detection_length, device=Dvc)
        upper = howmany
        if upper > detection_length:
            upper = detection_length
        indices_pre = permutation[0:upper]
        indices = ivector[indices_pre]
        #
        assert detection[indices].all(0)
        #
        for i in range(upper):
            indexi = indices[i]
            lsi = self.localscores[indexi]
            print(numpr(lsi, 2))
        return

    def predictedscoreLocal(self, M, ivector, xvector, yvector):
        #
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        #
        length = self.Examples["length"]
        prod = self.Examples["prod"]
        #
        availablexy = self.rr1.availablexy(length, prod).view(length, a, a)
        #
        assert availablexy[ivector, xvector, yvector].all(0)
        #
        Data = self.rr1.indexselectdata(self.Examples, ivector)
        dlength = Data["length"]
        dlrange = arangeic(dlength)
        #
        NoiseData = self.noise(Data)
        #
        pre_score = M.network2(NoiseData)
        #
        dlrangevxa = dlrange.view(dlength, 1).expand(dlength, a)
        arangevx = arangeic(a).view(1, a).expand(dlength, a)
        #
        predictedscore = pre_score[dlrange, xvector, yvector]
        return predictedscore

    def adapt_local_scores(self, ivector, xvector, yvector):
        #
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        ExamplePool = self.Examples
        #
        length = len(ivector)
        prod = (ExamplePool["prod"])[ivector]
        availablexyv = self.rr1.availablexy(length, prod).reshape(
            length * a * a
        )
        #
        available_count = (
            availablexyv.view(length, a * a).to(torch.int64).sum(1)
        )
        available_count_xf = (
            available_count.view(length, 1)
            .expand(length, a * a)
            .to(torch.float)
        )
        available_count_xf -= 1.0
        available_count_xf = torch.clamp(available_count_xf, 1.0, 100.0)
        #
        score = self.localscores[ivector].reshape(length * a * a)
        score[~availablexyv] = 100.0
        values, indices = torch.sort(score.view(length, a * a), 1)
        #
        position = torch.zeros((length, a * a), dtype=torch.float, device=Dvc)
        #
        lrangevx = arangeic(length).view(length, 1).expand(length, a * a)
        a2rangevx = arangeic(a * a).view(1, a * a).expand(length, a * a)
        position[lrangevx, indices] = (
            a2rangevx.to(torch.float) / available_count_xf
        )
        #
        position_score = position + score.view(length, a * a)
        #
        adapted_score = position_score.view(length, a, a)[
            arangeic(length), xvector, yvector
        ]
        return adapted_score

    def selectminibatchLocal(self, minibatchsize):
        #
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        ExamplePool = self.Examples
        xplength = ExamplePool["length"]
        if xplength == 0:
            return False, None, None, None, None
        xpprod = ExamplePool["prod"]
        xp_availablexy = self.rr1.availablexy(xplength, xpprod).view(
            xplength, a, a
        )
        #
        irangevxr = (
            arangeic(xplength)
            .view(xplength, 1, 1)
            .expand(xplength, a, a)
            .reshape(xplength * a * a)
        )
        xrangevxr = (
            arangeic(a)
            .view(1, a, 1)
            .expand(xplength, a, a)
            .reshape(xplength * a * a)
        )
        yrangevxr = (
            arangeic(a)
            .view(1, 1, a)
            .expand(xplength, a, a)
            .reshape(xplength * a * a)
        )
        #
        avdetect = xp_availablexy[irangevxr, xrangevxr, yrangevxr]
        #
        ivector_all = irangevxr[avdetect]
        xvector_all = xrangevxr[avdetect]
        yvector_all = yrangevxr[avdetect]
        #
        avlength = avdetect.to(torch.int64).sum(0)
        #
        permutation = torch.randperm(avlength, device=Dvc)
        mblength = minibatchsize
        if mblength > avlength:
            mblength = avlength
        indices = permutation[0:mblength]
        #
        ivector = ivector_all[indices]
        xvector = xvector_all[indices]
        yvector = yvector_all[indices]
        #
        scorebatch = self.adapt_local_scores(ivector, xvector, yvector)
        #
        return True, ivector, xvector, yvector, scorebatch

    def trainingLocal(
        self,
        M,
        numberofbatches,
        iterationsperbatch,
        style,
        minibatchsize,
        partname,
    ):
        #
        a = self.alpha
        #
        if style != "score-A" and style != "score-B" and style != "score-C":
            raise CoherenceError(
                "only allowed styles are score-A or score-B or score-C"
            )
        #
        if self.trainingprint:
            print(
                "/",
                style,
                numberofbatches,
                iterationsperbatch,
                partname,
                "/",
                end=" ",
            )
        for s in range(numberofbatches):
            (
                smb,
                ivector,
                xvector,
                yvector,
                scorebatch,
            ) = self.selectminibatchLocal(minibatchsize)
            if not smb:
                print("exit training")
                return
            #
            for i in range(iterationsperbatch):
                #
                M.optimizer2.zero_grad()
                #
                predictedscore = self.predictedscoreLocal(
                    M, ivector, xvector, yvector
                )
                #
                if style == "score-C":
                    lossA = M.criterionA(predictedscore, scorebatch)
                    lossB = M.criterionB(predictedscore, scorebatch)
                    loss = (lossA + lossB) / 2.0
                loss.backward()
                M.optimizer2.step()
            #
        #
        print("-", end=" ")
        #
        return

    def printlossaftertrainingLocal(self, M, minibatchsize, topicture):
        #
        a = M.pp.alpha
        #
        smb, ivector, xvector, yvector, scorebatch = self.selectminibatchLocal(
            minibatchsize
        )
        if not smb:
            print("data too small")
            return
        #
        mblength = len(ivector)
        #
        predictedscore = self.predictedscoreLocal(M, ivector, xvector, yvector)
        #
        lossa = M.criterionA(predictedscore, scorebatch)
        lra = numpr(lossa, 3)
        #
        lossb = M.criterionB(predictedscore, scorebatch)
        lrb = numpr(lossb, 3)
        #
        lossa_detach = lossa.detach()
        M.average_local_loss = (
            0.9 * M.average_local_loss + 0.1 * lossa_detach
        )  # change the variable name later!
        #
        print(
            "on ",
            itp(mblength),
            "values network -- L1 loss",
            lra,
            "MSE loss",
            lrb,
        )
        #
        if topicture:
            #
            self.HST.record_loss("local", lossa, lossb)
            #
            print("average local loss", numpr(M.average_local_loss, 4))
            #
            dotsize = torch.zeros((mblength), dtype=torch.int, device=Dvc)
            dotsize[:] = 2
            dotsize_np = nump(dotsize)
            #
            calcscore_npr = numpr(scorebatch, 3)
            predscore_npr = numpr(predictedscore, 3)
            #
            scoremax, index = torch.max(scorebatch, 0)
            linelimit = numpr(scoremax, 1)
            #
            plt.clf()
            plt.scatter(calcscore_npr, predscore_npr, dotsize_np)
            #
            # linelimit = 1.0
            plt.plot([0.0, linelimit], [0.0, 0.0], "g-", lw=1)
            plt.plot([0.0, 0.0], [0.0, linelimit], "g-", lw=1)
            plt.plot([0.0, linelimit], [0.0, linelimit], "r-", lw=1)
            #
            plt.show()
        return

    def learningLocal(self, M, globaliterations):
        #
        self.printlossaftertrainingLocal(M, 500, True)
        #
        tweak_cursor = self.HST.local_tweak_cursor
        tdensity = self.pp.tweak_density * (
            self.pp.tweak_decay ** tweak_cursor
        )
        tepsilon = self.pp.tweak_epsilon * (
            self.pp.tweak_decay ** tweak_cursor
        )
        self.HST.local_tweak_cursor += 1
        print(
            "tweaking local network at cursor",
            itp(tweak_cursor),
            "with density",
            numpr(tdensity, 4),
            "and epsilon",
            numpr(tepsilon, 4),
        )
        M.tweak_network(M.network2, tdensity, tepsilon)
        self.printlossaftertrainingLocal(M, 500, True)
        print("training", end=" ")
        #
        explore_pre_length = self.ExplorePrePool["length"]
        example_pre_length = self.ExamplesPrePool["length"]
        example_length = self.Examples["length"]
        self.HST.record_training(
            "local",
            globaliterations,
            explore_pre_length,
            example_pre_length,
            example_length,
        )
        #
        for g in range(globaliterations):
            print("/", end=" ")
            self.trainingLocal(M, 3, 20, "score-C", 20, "mb20")
            self.trainingLocal(M, 1, 15, "score-C", 30, "mb30")
            self.trainingLocal(M, 3, 4, "score-C", 60, "mb60")
            self.trainingLocal(M, 3, 10, "score-C", 40, "mb40")
            self.trainingLocal(M, 3, 3, "score-C", 20, "mb20")
            self.trainingLocal(M, 3, 2, "score-C", 40, "mb40")
            self.trainingLocal(M, 3, 1, "score-C", 30, "mb30")
            #
            print(" ")
            self.printlossaftertrainingLocal(M, 300, False)
            print("  ")
        self.printlossaftertrainingLocal(M, 500, True)
        print("=================    end score training   =================")
        return
