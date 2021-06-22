import gc

import torch

from classifier import Classifier
from constants import Dvc
from historical import Historical
from learner import Learner
from relations_4 import Relations4
from symmetric_group import SymmetricGroup
from utils import (
    CoherenceError,
    arangeic,
    itp,
    memReport,
    nump,
    numpi,
    numpr,
    zbinary,
)


class Driver:  # to run everything, it includes the sieve for instances sigma
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
        self.zbinatable = self.makezbinatable()
        #
        self.init_length, self.init_left_table = self.make_init_left_table()
        #
        self.donecount_collection = 0
        self.ECN_collection = 0.0
        self.ECN_average = 0.0
        #
        self.Cc = Classifier(self.Pp, HST)
        #
        self.Ll = Learner(self.rr4, HST)
        #
        self.HST = HST
        self.HST.record_driver(self.alpha, self.beta)

    def printprod(self, prod, loc):
        #
        a = self.alpha
        a2 = a * a
        a3 = a * a * a
        a3z = a3 + 1
        b = self.beta
        bz = b + 1
        #
        prodi = prod[loc]
        prarray = torch.zeros((a, a), dtype=torch.int64, device=Dvc)
        for x in range(a):
            for y in range(a):
                column = prod[loc, x, y]
                prarray[x, y] = self.printcolumn(column)
        print(nump(prarray))
        return

    def printcolumn(self, column):
        #
        a = self.alpha
        a2 = a * a
        a3 = a * a * a
        a3z = a3 + 1
        b = self.beta
        bz = b + 1
        #
        assert len(column) == bz
        #
        column_sum = column.to(torch.int64).sum(0)
        #
        if column_sum == 0:
            return -8
        #
        if column_sum == 1:
            value, prvalue = torch.max(column.to(torch.int64), 0)
            return prvalue
        if column_sum == bz:
            return -1
        exponents = arangeic(bz)
        powers = 10 ** exponents
        prvalue = (column.to(torch.int64) * powers).sum(0)
        prvalue += 3 * (10 ** bz)
        return prvalue

    def printcolumn2(self, column):
        #
        assert len(column) == 2
        #
        column_sum = column.to(torch.int64).sum(0)
        #
        if column_sum == 0:
            return -8
        if column_sum == 2:
            return 3
        #
        return column[1]

    def printleft(self, left, loc):
        #
        a = self.alpha
        a2 = a * a
        a3 = a * a * a
        a3z = a3 + 1
        b = self.beta
        bz = b + 1
        #
        prarray = torch.zeros((a, bz), dtype=torch.int64, device=Dvc)
        for x in range(a):
            for y in range(bz):
                column = left[loc, x, y]
                prarray[x, y] = self.printcolumn2(column)
        print(nump(prarray))
        return

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
                prarray[x, y] = self.printcolumn2(column)
        print(nump(prarray))
        return

    def print_prod_left_right(self, Data, i, as_sigma):
        prod = Data["prod"]
        left = Data["left"]
        right = Data["right"]
        #
        print("---------------------------------------------------")
        print(
            "at instance",
            itp(as_sigma),
            "the prod, left and right are respectively",
        )
        self.printprod(prod, i)
        self.printleft(left, i)
        self.printright(right, i)
        print("---------------------------------------------------")
        return

    def print_just_left(self, Data, i, as_sigma):
        left = Data["left"]
        #
        print("---------------------------------------------------")
        print(
            "at instance", itp(as_sigma), "the left multiplication matrix is"
        )
        self.printleft(left, i)
        print("---------------------------------------------------")
        return

    ##########################################################

    def makezbinatable(self):
        #
        a = self.alpha
        #
        power = 2 ** a
        #
        zbatable = torch.zeros((power, a), dtype=torch.bool, device=Dvc)
        for z in range(power):
            zbatable[z, :] = zbinary(a, z)
        return zbatable

    def collection(
        self, m
    ):  # this doesn't work perfectly, it needs to be re-sieved afterwards
        #
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        power = 2 ** a
        gl = self.sga.gtlength
        #
        gtbin = self.sga.gtbinary
        #
        if m <= 0:
            print("m <= 0 not allowed")
            raise CoherenceError("exiting")
        if m == 1:
            zrangevx = arangeic(power).view(1, power).expand(gl, power)
            detection = (gtbin >= zrangevx).all(0)
            clength = detection.to(torch.int64).sum(0)
            collec = arangeic(power)[detection].view(clength, m)
            #
            zbatablev = self.zbinatable.view(power, 1, a)
            collec_bin = zbatablev[detection]
            #
            previous_possibilities = torch.ones(
                (clength, power), dtype=torch.bool, device=Dvc
            )
            #
            previous_subgroup = torch.ones(
                (clength, gl), dtype=torch.bool, device=Dvc
            )
        #
        else:
            (
                cp,
                collec_prev,
                collec_bin_prev,
                poss_prev,
                subgroup_prev,
            ) = self.collection(m - 1)
            #
            detection_prev = poss_prev.view(cp * power)
            #
            subgroup_prev_vxr = (
                subgroup_prev.view(cp, 1, gl)
                .expand(cp, power, gl)
                .reshape(cp * power, gl)
            )
            z_current = (
                arangeic(power)
                .view(1, power, 1)
                .expand(cp, power, gl)
                .reshape(cp * power, gl)
            )
            g_current = arangeic(gl).view(1, gl).expand(cp * power, gl)
            detection_current = (
                (~subgroup_prev_vxr)
                | (gtbin[g_current, z_current] >= z_current)
            ).all(1)
            #
            detection = detection_prev & detection_current
            #
            clength = detection.to(torch.int64).sum(0)
            collec_prev_next = (
                collec_prev.view(cp, 1, m - 1)
                .expand(cp, power, m - 1)
                .reshape(cp * power, m - 1)
            )
            zrangevxr = (
                arangeic(power)
                .view(1, power, 1)
                .expand(cp, power, 1)
                .reshape(cp * power, 1)
            )
            new_prev = collec_prev_next[detection]
            zvector = zrangevxr[detection]
            collec = torch.cat((new_prev, zvector), 1)
            #
            collec_bin_prev_next = (
                collec_bin_prev.view(cp, 1, m - 1, a)
                .expand(cp, power, m - 1, a)
                .reshape(cp * power, m - 1, a)
            )
            new_bin_prev = collec_bin_prev_next[detection]
            zbatablevxr = (
                self.zbinatable.view(1, power, 1, a)
                .expand(cp, power, 1, a)
                .reshape(cp * power, 1, a)
            )
            zbavector = zbatablevxr[detection]
            collec_bin = torch.cat((new_bin_prev, zbavector), 1)
            #
            poss_prev_vxr = (
                poss_prev.view(cp, 1, power)
                .expand(cp, power, power)
                .reshape(cp * power, power)
            )
            previous_possibilities = poss_prev_vxr[detection]
            #
            subgroup_prev_vxr = (
                subgroup_prev.view(cp, 1, gl)
                .expand(cp, power, gl)
                .reshape(cp * power, gl)
            )
            previous_subgroup = subgroup_prev_vxr[detection]
        #
        grange_vx = arangeic(gl).view(1, gl, 1).expand(clength, gl, m)
        collec_vx = collec.view(clength, 1, m).expand(clength, gl, m)
        transform = gtbin[grange_vx, collec_vx]
        transform_sort, t_indices = torch.sort(transform, 2)
        #
        subgroup = (collec_vx == transform_sort).all(2)
        #
        #
        next_transform = gtbin.view(1, gl, power).expand(clength, gl, power)
        prev_bound = (
            (collec[:, m - 1]).view(clength, 1, 1).expand(clength, gl, power)
        )
        prev_subgroupvx = previous_subgroup.view(clength, gl, 1).expand(
            clength, gl, power
        )
        #
        next_possib_prev = (
            (~prev_subgroupvx) | (prev_bound <= next_transform)
        ).all(1)
        possibilities = next_possib_prev & previous_possibilities
        #
        return clength, collec, collec_bin, possibilities, subgroup

    def collectiontest(self, amount):
        #
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        #
        clength, collec, collec_bin, possibilities, subgroup = self.collection(
            b
        )
        print("collection has length", itp(clength))
        upper = 20
        if upper > clength:
            upper = clength
        print("first", itp(upper), "elements are as follows")
        #
        collec_sum = (collec_bin.to(torch.int64).sum(2)).sum(1)
        collec_sum1 = collec_bin.to(torch.int64).sum(1)
        collec_sum2 = collec_bin.to(torch.int64).sum(2)
        for q in range(a * b + 1):
            freq = (collec_sum == q).to(torch.int64).sum(0)
            print("for amount", itp(q), "frequency", itp(freq))
        amount_detect = collec_sum == amount
        #
        ad_freq = amount_detect.to(torch.int64).sum(0)
        collec_detect = collec_bin[amount_detect]
        print("ad freq", itp(ad_freq))
        for i in range(ad_freq):
            print("--------------------------")
            print(numpi(collec_detect[i]))
            print("collec_sum1", nump((collec_sum1[amount_detect])[i]))
            print("collec_sum2", nump((collec_sum2[amount_detect])[i]))
        return

    def lex_lt(self, width, z1, z2):
        #
        assert width > 0
        #
        if width == 1:
            z1new = z1[:, 0]
            z2new = z2[:, 0]
            lt = z1new < z2new
            return lt
        z1prev = z1[:, 0 : width - 1]
        z2prev = z2[:, 0 : width - 1]
        #
        lt_prev = self.lex_lt(width - 1, z1prev, z2prev)
        #
        eq_prev = (z1prev == z2prev).all(1)
        #
        z1new = z1[:, width - 1]
        z2new = z2[:, width - 1]
        lt = lt_prev | (eq_prev & (z1new < z2new))
        return lt

    def collection_sieve(self):
        #
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        power = 2 ** a
        gl = self.sga.gtlength
        #
        gtbin = self.sga.gtbinary
        #
        clength, collec, collec_bin, possibilities, subgroup = self.collection(
            b
        )
        #
        collecvxr = (
            collec.view(clength, 1, b)
            .expand(clength, gl, b)
            .reshape(clength * gl, b)
        )
        grangevxr = (
            arangeic(gl)
            .view(1, gl, 1)
            .expand(clength, gl, b)
            .reshape(clength * gl, b)
        )
        #
        transform = gtbin[grangevxr, collecvxr]
        transform_sort, t_indices = torch.sort(transform, 1)
        #
        transform_ltv = self.lex_lt(b, transform_sort, collecvxr)
        transform_lt = transform_ltv.view(clength, gl)
        #
        throw = transform_lt.any(1)
        #
        detection = ~throw
        sieved_length = detection.to(torch.int64).sum(0)
        sieved_collection = collec[detection]
        sieved_collection_bin = collec_bin[detection]
        #
        return sieved_length, sieved_collection, sieved_collection_bin

    def sieve_test(self):
        (
            sieved_length,
            sieved_collection,
            sieved_collection_bin,
        ) = self.collection_sieve()
        print("sieved collection has length", itp(sieved_length))
        return

    def make_init_left_table(self):
        #
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        (
            length,
            sieved_collection,
            sieved_collection_bin,
        ) = self.collection_sieve()
        #
        init_left_table = torch.zeros(
            (length, a, bz, 2), dtype=torch.bool, device=Dvc
        )
        #
        left_value = sieved_collection_bin.permute(0, 2, 1)
        init_left_table[:, :, 0:b, 1] = left_value
        init_left_table[:, :, 0:b, 0] = ~left_value
        init_left_table[:, :, b, 0] = True
        #
        print(" ")
        print("available left table instances are 0 <= sigma <", itp(length))
        print(
            "   ---> these should be noted as the possible values of sigma for the proof instances"
        )
        #
        zerocolumn = (~left_value).all(1)
        zerocolumn_number = zerocolumn.to(torch.int64).sum(1)
        overhalf = 2 * zerocolumn_number > b
        overhalf_indices = arangeic(length)[overhalf]
        print("   ")
        print("locations with > half zero columns are", nump(overhalf_indices))
        print(
            "   ---> it is suggested not to use the sigma instances in this list, specially for larger cases "
        )
        print("   ")
        #
        return length, init_left_table

    ##########################################################

    def initialdata(self, instancevector, dropoutlimit):
        #
        a = self.alpha
        a2 = self.alpha2
        a3 = self.alpha3
        a3z = self.alpha3z
        b = self.beta
        bz = self.betaz
        #
        length = len(instancevector)
        prod = torch.ones((length, a, a, bz), dtype=torch.bool, device=Dvc)
        #
        left = self.init_left_table[instancevector]
        right = torch.ones((length, bz, a, 2), dtype=torch.bool, device=Dvc)
        right[:, b, :, 1] = False
        #
        if self.Pp.verbose:
            print("initial prod at 0")
            self.printprod(prod, 0)
            print("initial left at 0")
            print(numpi(left[0, :, :, 1]))
            print("initial right at 0")
            print(numpi(right[0, :, :, 1]))
        #
        depth = torch.zeros((length), dtype=torch.int, device=Dvc)
        #
        ternary = torch.ones(
            (length, a, a, a, 2), dtype=torch.bool, device=Dvc
        )
        #
        info = torch.zeros(
            (length, self.Pp.infosize), dtype=torch.int64, device=Dvc
        )
        info[:, self.Pp.sampleinfolower : self.Pp.sampleinfoupper] = -1
        #
        #
        RawData = {
            "length": length,
            "depth": depth,
            "prod": prod,
            "left": left,
            "right": right,
            "ternary": ternary,
            "info": info,
        }
        #
        if dropoutlimit > 0:
            rectangle = (
                arangeic(length)
                .view(length, 1)
                .expand(length, dropoutlimit)
                .reshape(length * dropoutlimit)
            )
            index_slice = (torch.randperm(length * dropoutlimit, device=Dvc))[
                0:dropoutlimit
            ]
            indices = rectangle[index_slice]
            AugmentedData = self.rr1.indexselectdata(RawData, indices)
            # set phase
            tirage = torch.rand((dropoutlimit), device=Dvc)
            phase1_upper = 0.8 * self.Pp.splitting_probability
            phase12_upper = 0.8
            phase1 = tirage < phase1_upper
            phase2 = (tirage < phase12_upper) & (~phase1)
            AugmentedData["info"][:, self.Pp.phase][phase2] = 2
            AugmentedData["info"][:, self.Pp.phase][phase1] = 1
            return AugmentedData
        return RawData

    def print_instances(self):
        #
        (
            instancevector,
            training_instances,
            proof_title,
        ) = self.instance_chooser()
        InitialData = self.initialdata(instancevector, 0)
        length = InitialData["length"]
        print(proof_title)
        print(
            "initial data from this collection of instances has length",
            itp(length),
        )
        for i in range(length):
            as_sigma = instancevector[i]
            # self.print_prod_left_right(InitialData,i,as_sigma)
            self.print_just_left(InitialData, i, as_sigma)
        print("- - - - - - - - - - - - - - - - - - -")
        return

    def classificationproof(
        self, Mstrat, Mlearn, dropoutlimit, proving_instances, title_text
    ):  # dropoutlimit =-1 for no dropout
        print("---   ---   ---   ---   ---   ---   ---   ---   ---")
        print("                classification proof")
        print("---   ---   ---   ---   ---   ---   ---   ---   ---")
        #
        self.HST.title_text_sigma_proof = title_text
        #
        self.rr4.proofnumber = 0
        self.rr4.proofinstance = 0
        self.rr4.allnumbers = 1
        #
        if dropoutlimit == 0:
            self.HST.reset_current_proof()
        #
        self.Cc.initialize()
        #
        InitialData = self.initialdata(proving_instances, dropoutlimit)
        #
        if dropoutlimit > 0:
            AssocInitialData = self.rr2.process(InitialData)
            activedetect, donedetect, impossibledetect = self.rr2.filterdata(
                AssocInitialData
            )
            ActiveInitialData = self.rr1.detectsubdata(
                AssocInitialData, activedetect
            )
            #
            explore_upper = self.Pp.root_injection
            if explore_upper > dropoutlimit:
                explore_upper = dropoutlimit
            self.Ll.prepoolExplore(Mlearn, ActiveInitialData, explore_upper)
        #
        pl, ActivePool, DonePool, prooflength = self.rr4.proofloop(
            Mstrat, Mlearn, self.Cc, InitialData, dropoutlimit
        )
        #
        print("---   ---   ---   ---   ---   ---   ---   ---   ---")
        print(
            "this proof was treating alpha =",
            itp(self.alpha),
            "beta =",
            itp(self.beta),
        )
        print("proof ended after step number", itp(prooflength))
        if ActivePool["length"] > 0:
            print(
                "proof outputs Active Data of length",
                itp(ActivePool["length"]),
            )
        print("proof has done count", itp(self.rr4.donecount), end=" ")
        print("and estimated cumulative nodes", numpr(self.rr4.ECN, 1))
        self.donecount_collection += self.rr4.donecount
        self.ECN_collection += self.rr4.ECN
        print("classifier eq pool has length", itp(self.Cc.eqlength))
        #
        if dropoutlimit == 0:
            if Mstrat.benchmark:
                self.HST.record_current_proof(self.Pp, benchmark=True)
            else:
                self.HST.record_current_proof(self.Pp)
        #
        print("---   ---   ---   ---   ---   ---   ---   ---   ---")
        print("          classification proof done")
        print("---   ---   ---   ---   ---   ---   ---   ---   ---")
        print(
            "===   ===   ===   ===   ===   ===   ===   ===   ===   ===   ===   ==="
        )
        #
        return

    #### mini programs for creation of the instancevector_title object (it is really a pair)

    def InAll(self):
        instance_vector = arangeic(self.init_length)
        #
        title_text = f"for all sigma instances"
        #
        return instance_vector, instance_vector, title_text

    def InOne(self, instance):
        assert 0 <= instance < self.init_length
        instance_vector = torch.zeros((1), dtype=torch.int64, device=Dvc)
        instance_vector[0] = instance
        #
        title_text = f"for sigma instance {instance}"
        #
        return instance_vector, instance_vector, title_text

    def InSeg(self, lower, upper):
        assert 0 <= lower < self.init_length
        assert lower < upper
        if upper > self.init_length:
            print("retracting the upper value to", self.init_length)
        upper_mod = upper
        if upper_mod > self.init_length:
            upper_mod = self.init_length
        instance_vector = arangeic(self.init_length)[lower:upper]
        #
        title_text = f"for sigma instances in range {lower}:{upper_mod}"
        #
        return instance_vector, instance_vector, title_text

    def InSkip(self, skip, lower, upper):
        assert 0 <= lower < self.init_length
        assert lower < upper
        upper_mod = upper
        if upper_mod > self.init_length:
            upper_mod = self.init_length
        detection = torch.zeros(
            (self.init_length), dtype=torch.bool, device=Dvc
        )
        detection[lower:upper] = True
        detection[skip] = False
        training_vector = arangeic(self.init_length)[detection]
        #
        proving_vector = torch.zeros((1), dtype=torch.int64, device=Dvc)
        proving_vector[0] = skip
        #
        title_text = f"training for sigma in range {lower}:{upper} skipping and proving {skip}"
        #
        return proving_vector, training_vector, title_text

    def InList(self, proving_list, training_list):
        proving_vector = torch.tensor(
            proving_list, dtype=torch.int64, device=Dvc
        )
        training_vector = torch.tensor(
            training_list, dtype=torch.int64, device=Dvc
        )
        #
        title_text = f"training for sigma instances in list {training_list} and proving {proving_list}"
        #
        return proving_vector, training_vector, title_text

    def basicloop(self, Mstrat, Mlearn, training_instances, title_text):
        #
        dropout2 = 300
        #
        self.HST.title_text_sigma_train = title_text
        #
        for i in range(self.Pp.basicloop_iterations):
            print(
                "------      ------      basic loop",
                itp(i),
                "------      ------",
            )
            #
            #
            self.Pp.dropout_style = "regular"
            self.classificationproof(
                Mstrat, Mlearn, dropout2, training_instances, title_text
            )
            self.Ll.prepoolSamples(
                Mlearn, self.rr4.SamplePool, self.Ll.new_examples_max
            )
            #
            self.Ll.scoreExamples(Mlearn)
            #
            self.Pp.dropout_style = "adaptive"
            #
            self.classificationproof(
                Mstrat, Mlearn, dropout2, training_instances, title_text
            )
            self.Ll.prepoolSamples(
                Mlearn, self.rr4.SamplePool, self.Ll.new_examples_max
            )
            self.classificationproof(
                Mstrat, Mlearn, 100, training_instances, title_text
            )
            self.Ll.addscoredexamples(Mlearn)
            self.Ll.prepoolSamples(
                Mlearn, self.rr4.SamplePool, self.Ll.new_examples_max
            )
            #
            self.Ll.scoreExamples(Mlearn)
            self.Ll.scoreExplore(Mlearn)
            self.Ll.scoreOutlier(Mlearn)
            print(
                "------  global learning ---   ----",
                itp(i),
                "------      ------",
            )
            self.Ll.learningGlobal(
                Mlearn, self.Pp.basicloop_training_iterations
            )
            if Mlearn.network2_trainable:
                print(
                    "------  local learning  ---   ----",
                    itp(i),
                    "------      ------",
                )
                self.Ll.learningLocal(
                    Mlearn, self.Pp.basicloop_training_iterations
                )
            else:
                print("network 2 is not trainable")
            print(
                "======      ======      ======      ======      ======      ======"
            )
            gcc = gc.collect()
            memReport("mg")
            print(
                "======      ======      ======      ======      ======      ======"
            )
        print(
            "end of",
            itp(self.Pp.basicloop_iterations),
            "iterations of the basic loop",
        )
        print(
            "======      ======      ======      ======      ======      ======"
        )
        return

    def basicloop_classificationproof(
        self, Mstrat, Mlearn, proving_instances, training_instances, title_text
    ):
        #
        print("suggested number of proof cycles: between 20 and 50")
        runs = int(input("input the number of proof cycles to do : "))
        for s in range(runs):
            self.basicloop(Mstrat, Mlearn, training_instances, title_text)
            print(">>>", s + 1, "   (out of ", runs, " )")
            self.classificationproof(
                Mstrat, Mlearn, 0, proving_instances, title_text
            )
            self.HST.graph_history(self.Pp, "big")
        return

    #### the following function automates the process of choosing a collection of instances to do

    def instance_chooser(self):
        print(
            "choose instances, this chooser allows : all, one, seg, skip  (do by hand for list input---see optional cells below)"
        )
        instance_type = input("input type : ")
        if (
            instance_type != "all"
            and instance_type != "one"
            and instance_type != "seg"
            and instance_type != "skip"
        ):
            print("please use one of : all one seg skip")
            raise CoherenceError("exiting")
        if instance_type == "all":
            print(
                f"this will do all sigma instances in the existing range 0 <= sigma < {self.init_length}"
            )
            proving_instances, training_instances, title_text = self.InAll()
        if instance_type == "one":
            print(
                f"to do a single sigma instance, choose in range 0 <= sigma < {self.init_length}"
            )
            sigma_instance = int(input("input sigma instance : "))
            proving_instances, training_instances, title_text = self.InOne(
                sigma_instance
            )
        if instance_type == "seg":
            print("to do sigma instances in a segment lower <= sigma < upper")
            segment_lower = int(input("input lower : "))
            segment_upper = int(input("input upper : "))
            proving_instances, training_instances, title_text = self.InSeg(
                segment_lower, segment_upper
            )
        if instance_type == "skip":
            print(
                "to train on a segment skipping a single value, and do proofs of that value"
            )
            segment_lower = int(input("input training segment lower : "))
            segment_upper = int(input("input training segment upper : "))
            skip = int(
                input(
                    "input instance to prove, and to skip during training : "
                )
            )
            proving_instances, training_instances, title_text = self.InSkip(
                skip, segment_lower, segment_upper
            )
        return proving_instances, training_instances, title_text
