import matplotlib.pyplot as plt
import torch

from constants import Dvc, torch_pi
from utils import CoherenceError, itp, nump, numpr


class Historical:
    def __init__(self, hlength_max):
        self.hlength_max = hlength_max
        self.hwidth = 20
        self.hfwidth = 10
        self.prwidth = 20
        #
        self.hlength = 0
        #
        self.histi = torch.zeros(
            (self.hlength_max, self.hwidth), dtype=torch.int64, device=Dvc
        )
        self.histf = torch.zeros(
            (self.hlength_max, self.hfwidth), dtype=torch.float, device=Dvc
        )
        #
        self.proofrecord = torch.zeros(
            (self.hlength_max, self.prwidth), dtype=torch.int64, device=Dvc
        )
        self.prcursor = 0
        self.local_tweak_cursor = 0
        self.global_tweak_cursor = 0
        #
        self.current_proof_valency_frequency = torch.zeros(
            (10), dtype=torch.int64, device=Dvc
        )
        self.current_proof_impossible_count = 0
        self.current_proof_done_count = 0
        self.current_proof_passive_count = 0
        self.current_proof_benchmark = 0
        #
        self.title_text_sigma_proof = None
        self.title_text_sigma_train = None
        #
        self.training_counter = 0
        #
        self.proof_nodes_max = 200000
        #
        self.D = {
            "Global": 0,
            "Local": 1,
            "LocalCE": 2,
            "Benchmark": 3,
            "Regular": 4,
            "Adaptive": 5,
            "Uniform": 6,
            "Parameters": 7,
            "Driver": 8,
            "Model": 9,
            "Loss": 10,
            "Training": 11,
            "BenchmarkProof": 12,
            "FullProof": 13,
            "DropoutProof": 14,
        }

    def reset_current_proof(self):
        self.current_proof_valency_frequency[:] = 0
        self.current_proof_impossible_count = 0
        self.current_proof_done_count = 0
        self.current_proof_passive_count = 0
        self.current_proof_benchmark = 0
        return

    def record_current_proof(self, benchmark=False):
        if self.prcursor >= self.hlength_max:
            print("proof data recording overflow")
            return
        self.proofrecord[
            self.prcursor, 0
        ] = self.current_proof_impossible_count
        self.proofrecord[self.prcursor, 1] = self.current_proof_done_count
        self.proofrecord[self.prcursor, 2] = self.current_proof_passive_count
        self.proofrecord[
            self.prcursor, 3:13
        ] = self.current_proof_valency_frequency
        self.proofrecord[self.prcursor, 14] = 0
        if benchmark:
            self.proofrecord[self.prcursor, 14] = 1
            print("recorded benchmark proof", end=" ")
        else:
            print("recorded full proof", end=" ")
        self.print_proof_recordi(self.prcursor, Pp)
        self.prcursor += 1
        return

    def print_proof_recordi(self, i, Pp):
        #
        upperval = Pp.beta + 2
        if upperval > 10:
            upperval = 10
        impcount = itp(self.proofrecord[i, 0])
        donecount = itp(self.proofrecord[i, 1])
        passivecount = itp(self.proofrecord[i, 2])
        valency = nump(self.proofrecord[i, 3 : 3 + upperval])
        #
        print(
            ". done",
            donecount,
            "impossible",
            impcount,
            "passive",
            passivecount,
            "by valency",
            valency,
        )
        return

    def print_proof_records(self, Pp):
        #
        ###
        if Pp.profile_filter_on:
            prof_filt = "on"
        else:
            prof_filt = "off"
        if Pp.halfones_filter_on:
            ho_filt = "on"
        else:
            ho_filt = "off"
        bl_iter = Pp.basicloop_iterations
        bl_train = Pp.basicloop_training_iterations
        #
        global_p = itp(Pp.global_params)
        local_p = itp(Pp.local_params)
        ###
        #
        print("-------------------------------------------------------------")
        print(
            "proof records for a,b =",
            itp(Pp.alpha),
            itp(Pp.beta),
            "model with n=",
            Pp.model_n,
            self.title_text_sigma_proof,
        )
        print(
            f"with {bl_iter} basic loops per proof and {bl_train} training segments per basic loop, profile filter {prof_filt}, halfones filter {ho_filt}"
        )
        print(
            f"global model has {global_p} and local rank+score model has {local_p} trainable parameters"
        )
        print("-------------------------------------------------------------")
        proof_number = 0
        for i in range(self.prcursor):
            if self.proofrecord[i, 14] > 0:
                print("benchmark proof", end=" ")
            else:
                print("proof", itp(proof_number), end=" ")
                proof_number += 1
            self.print_proof_recordi(i, Pp)
        print("-------------------------------------------------------------")
        return

    def noiselevel(self, P, count_tensor):
        counterf = count_tensor.to(torch.float)
        counter_period_units = counterf / P.noise_period
        phase = counter_period_units * 2 * torch_pi
        one_plus_cos_over_two = (torch.cos(phase) + 1.0) / 2
        decay = P.noise_decay ** counter_period_units
        level = P.noise_level * one_plus_cos_over_two * decay
        return level

    def reset(self):
        self.histi[:, :] = 0
        self.histf[:, :] = 0.0
        self.hlength = 0
        self.training_counter = 0
        self.prcursor = 0
        self.local_tweak_cursor = 0
        self.global_tweak_cursor = 0
        print("reinitialized history")
        return

    def increment(self):
        if self.hlength >= self.hlength_max:
            print("Historical at maximum length, can't add any new entries")
            raise CoherenceError("exiting")
        cursor = self.hlength
        self.hlength += 1
        assert 0 <= cursor < self.hlength_max
        return cursor

    def record_parameters(self, alpha, beta):
        cursor = self.increment()
        self.histi[cursor, 0] = self.D["Parameters"]
        self.histi[cursor, 1] = alpha
        self.histi[cursor, 2] = beta
        return

    def record_driver(self, alpha, beta):
        cursor = self.increment()
        self.histi[cursor, 0] = self.D["Driver"]
        self.histi[cursor, 1] = alpha
        self.histi[cursor, 2] = beta
        return

    def record_model(self, n):
        cursor = self.increment()
        self.histi[cursor, 0] = self.D["Model"]
        self.histi[cursor, 1] = n
        return

    def record_loss(self, style, L1_loss, MSE_loss):
        cursor = self.increment()
        self.histi[cursor, 0] = self.D["Loss"]
        if style != "global" and style != "local" and style != "local_ce":
            raise CoherenceError("unsupported style in record_loss")
        if style == "global":
            self.histi[cursor, 1] = self.D["Global"]
        if style == "local":
            self.histi[cursor, 1] = self.D["Local"]
        if style == "local_ce":
            self.histi[cursor, 1] = self.D["LocalCE"]
        if torch.is_tensor(MSE_loss):
            MSE_loss_detach = MSE_loss.detach()
        else:
            MSE_loss_detach = MSE_loss
        self.histf[cursor, 0] = L1_loss.detach()
        self.histf[cursor, 1] = MSE_loss_detach
        #
        self.training_counter += 1
        return

    def record_training(
        self,
        style,
        iterations,
        explore_pre_pool,
        example_pre_pool,
        example_pool,
    ):
        cursor = self.increment()
        #
        self.histi[cursor, 0] = self.D["Training"]
        if style != "global" and style != "local":
            raise CoherenceError("unsupported style in record_training")
        if style == "global":
            self.histi[cursor, 1] = self.D["Global"]
        if style == "local":
            self.histi[cursor, 1] = self.D["Local"]
        self.histi[cursor, 2] = iterations
        self.histi[cursor, 3] = explore_pre_pool
        self.histi[cursor, 4] = example_pre_pool
        self.histi[cursor, 5] = example_pool
        return

    def record_full_proof(self, M, steps, cumulative_nodes, done_nodes):
        cursor = self.increment()
        #
        if M.benchmark:
            self.histi[cursor, 0] = self.D["BenchmarkProof"]
        else:
            self.histi[cursor, 0] = self.D["FullProof"]
        self.histi[cursor, 1] = steps
        self.histi[cursor, 2] = cumulative_nodes
        self.histi[cursor, 3] = done_nodes
        return

    def record_dropout_proof(self, style, dropout, steps, ECN):
        cursor = self.increment()
        #
        if style != "regular" and style != "adaptive" and style != "uniform":
            raise CoherenceError("unsupported style in record_dropout_proof")
        #
        self.histi[cursor, 0] = self.D["DropoutProof"]
        if style == "regular":
            self.histi[cursor, 1] = self.D["Regular"]
        if style == "adaptive":
            self.histi[cursor, 1] = self.D["Adaptive"]
        if style == "uniform":
            self.histi[cursor, 1] = self.D["Uniform"]
        self.histi[cursor, 2] = dropout
        self.histi[cursor, 3] = steps
        ecnr = torch.round(ECN).to(torch.int64)
        self.histi[cursor, 4] = ecnr
        return

    def print_history(self):
        length = self.hlength
        print("--  --  --  --  --  --  --  --  --  --  --  --  --  --  --")
        print("     printing history of length", itp(length))
        print("--  --  --  --  --  --  --  --  --  --  --  --  --  --  --")
        for cursor in range(length):
            tag = self.histi[cursor, 0]
            a = itp(self.histi[cursor, 1])
            b = itp(self.histi[cursor, 2])
            c = itp(self.histi[cursor, 3])
            d = itp(self.histi[cursor, 4])
            e = itp(self.histi[cursor, 5])
            f = itp(self.histi[cursor, 6])
            #
            x = numpr(self.histf[cursor, 0], 3)
            y = numpr(self.histf[cursor, 1], 3)
            #
            print("(", cursor, ")--", end="")
            #
            if tag == self.D["Parameters"]:
                print("setting parameters for alpha=", a, "beta=", b)
            #
            if tag == self.D["Driver"]:
                print("initialize driver for alpha=", a, "beta=", b)
            #
            if tag == self.D["Model"]:
                print("initialize model with n =", a)
            #
            if tag == self.D["Loss"]:
                if a == self.D["Global"]:
                    print("test global model, L1 loss", x, "MSE loss", y)
                if a == self.D["Local"]:
                    print("test local model, L1 loss", x, "MSE loss", y)
                if a == self.D["LocalCE"]:
                    print("test local model, CE loss", x)
            #
            if tag == self.D["Training"]:
                if a == self.D["Global"]:
                    print("training global model", end=" ")
                if a == self.D["Local"]:
                    print("training local model", end=" ")
                print(
                    "iterations",
                    b,
                    "explore prepool",
                    c,
                    "example prepool",
                    d,
                    "example_pool",
                    e,
                )
            #
            if tag == self.D["BenchmarkProof"]:
                print(
                    "BENCHMARK PROOF in",
                    a,
                    "steps",
                    b,
                    "cumulative nodes",
                    c,
                    "done leaves",
                )
            #
            if tag == self.D["FullProof"]:
                print(
                    "FULL PROOF in",
                    a,
                    "steps",
                    b,
                    "cumulative nodes",
                    c,
                    "done leaves",
                )
            #
            if tag == self.D["DropoutProof"]:
                print("proof with dropout style", end="")
                if a == self.D["Regular"]:
                    print(" regular ", end="")
                if a == self.D["Adaptive"]:
                    print(" adaptive ", end="")
                if a == self.D["Uniform"]:
                    print(" uniform ", end="")
                print(
                    "threshold",
                    b,
                    "in",
                    c,
                    "steps with estimated cumulative nodes",
                    d,
                )
        #
        print("--  --  --  --  --  --  --  --  --  --  --  --  --  --  --")
        print("     end printing history of length", itp(length))
        print("--  --  --  --  --  --  --  --  --  --  --  --  --  --  --")
        #
        return

    def graph_history(self, P, style):
        #
        alpha = P.alpha
        beta = P.beta
        nu = P.model_n
        #
        ###
        if P.profile_filter_on:
            prof_filt = "on"
        else:
            prof_filt = "off"
        if P.halfones_filter_on:
            ho_filt = "on"
        else:
            ho_filt = "off"
        bl_iter = P.basicloop_iterations
        bl_train = P.basicloop_training_iterations
        #
        global_p = itp(P.global_params)
        local_p = itp(P.local_params)
        ###
        plotcount = 0
        for cursor in range(self.hlength):
            tag = self.histi[cursor, 0]
            if tag == self.D["FullProof"]:
                plotcount += 1
        baseline = 0
        ecn_graph = torch.zeros((plotcount), dtype=torch.int64, device=Dvc)
        attempts = arangeic(plotcount)
        attempt_number = 0
        for cursor in range(self.hlength):
            tag = self.histi[cursor, 0]
            b = self.histi[cursor, 2]
            if tag == self.D["BenchmarkProof"]:
                baseline = b
            if tag == self.D["FullProof"]:
                ecn_graph[attempt_number] = b
                attempt_number += 1
        ecn_graph = torch.clamp(ecn_graph, 0, self.proof_nodes_max)
        plt.clf()
        if style == "big":
            plt.figure(figsize=(17, 8))
        else:
            plt.figure(figsize=(8, 4))
        ###
        phrase1 = f"Proofs for a= {alpha}, b={beta} for model with n= {nu} and heuristic baseline, {HST.title_text_sigma_proof}"
        phrase2 = f"with {bl_iter} basic loops per proof and {bl_train} training segments per basic loop, profile filter {prof_filt}, halfones filter {ho_filt}"
        phrase3 = f"global model has {global_p} and local rank+score model has {local_p} trainable parameters"
        plt.title(phrase1 + "\n" + phrase2 + "\n" + phrase3)
        ###
        plt.xlabel("number of training rounds")
        plt.ylabel("cumulative nodes")
        plt.plot(
            [0, plotcount - 1], [itp(baseline), itp(baseline)], "deepskyblue"
        )
        plt.plot(nump(attempts), nump(ecn_graph), ".-")
        plt.show()
        #
        pcg = 0
        pcl = 0
        for cursor in range(self.hlength):
            tag = self.histi[cursor, 0]
            a = self.histi[cursor, 1]
            if tag == self.D["Loss"]:
                if a == self.D["Global"]:
                    pcg += 1
                if a == self.D["Local"]:
                    pcl += 1
                if a == self.D["LocalCE"]:
                    pcl += 1
        L1graph_local = torch.zeros((pcl), dtype=torch.float, device=Dvc)
        MSEgraph_local = torch.zeros((pcl), dtype=torch.float, device=Dvc)
        CEgraph_local = torch.zeros((pcl), dtype=torch.float, device=Dvc)
        L1graph_global = torch.zeros((pcg), dtype=torch.float, device=Dvc)
        MSEgraph_global = torch.zeros((pcg), dtype=torch.float, device=Dvc)
        measurements_local = arangeic(pcl)
        measurements_global = arangeic(pcg)
        measurement_number_local = 0
        measurement_number_global = 0
        for cursor in range(self.hlength):
            tag = self.histi[cursor, 0]
            a = self.histi[cursor, 1]
            x = self.histf[cursor, 0]
            y = self.histf[cursor, 1]
            if tag == self.D["Loss"]:
                if a == self.D["Global"]:
                    L1graph_global[measurement_number_global] = x
                    MSEgraph_global[measurement_number_global] = y
                    measurement_number_global += 1
                if a == self.D["Local"]:
                    L1graph_local[measurement_number_local] = x
                    MSEgraph_local[measurement_number_local] = y
                    measurement_number_local += 1
                if a == self.D["LocalCE"]:
                    CEgraph_local[measurement_number_local] = x
                    measurement_number_local += 1
        L1avg_global = (
            L1graph_global[0 : pcg - 2]
            + L1graph_global[1 : pcg - 1]
            + L1graph_global[2:pcg]
        ) / 3.0
        L1avg_global = torch.clamp(L1avg_global, 0.0, 0.2)
        L1avg_local = (
            L1graph_local[0 : pcl - 2]
            + L1graph_local[1 : pcl - 1]
            + L1graph_local[2:pcl]
        ) / 3.0
        #
        L1avg_local_red = (
            L1avg_local / 2.0
        )  # so it fits on the graph better, with the current adaptive_score function
        #
        L1avg_local_red = torch.clamp(L1avg_local_red, 0.0, 0.2)
        MSEavg_global = (
            MSEgraph_global[0 : pcg - 2]
            + MSEgraph_global[1 : pcg - 1]
            + MSEgraph_global[2:pcg]
        ) / 3.0
        MSEavg_global = torch.clamp(MSEavg_global, 0.0, 0.2)
        MSEavg_local = (
            MSEgraph_local[0 : pcl - 2]
            + MSEgraph_local[1 : pcl - 1]
            + MSEgraph_local[2:pcl]
        ) / 3.0
        MSEavg_local = torch.clamp(MSEavg_local, 0.0, 0.2)
        CEavg_local = (
            CEgraph_local[0 : pcl - 2]
            + CEgraph_local[1 : pcl - 1]
            + CEgraph_local[2:pcl]
        ) / 3.0
        CEavg_local = CEavg_local / 10.0
        CEavg_local = torch.clamp(CEavg_local, 0.0, 0.2)
        #
        #
        noiselevel = HST.noiselevel(P, measurements_global)
        #
        plt.clf()
        if style == "big":
            plt.figure(figsize=(17, 8))
        else:
            plt.figure(figsize=(8, 4))
        ###
        phrase1b = f"Training for a= {alpha}, b={beta} for model with n= {nu}, {HST.title_text_sigma_train}"
        # phrase 2 is the same as before
        plt.title(phrase1b + "\n" + phrase2 + "\n" + phrase3)
        ###
        plt.xlabel("training")
        plt.ylabel("loss")
        #
        plt.plot(
            nump(measurements_local[5:pcl]),
            numpr(L1avg_local_red[3 : pcl - 2], 4),
            label="local-L1/2",
        )
        plt.plot(
            nump(measurements_global[5:pcg]),
            numpr(L1avg_global[3 : pcg - 2], 4),
            label="global-L1",
        )
        #
        plt.plot(
            nump(measurements_local[5:pcl]),
            numpr(MSEavg_local[3 : pcl - 2], 5),
            label="local-MSE",
        )
        plt.plot(
            nump(measurements_global[5:pcg]),
            numpr(MSEavg_global[3 : pcg - 2], 5),
            label="global-MSE",
        )
        #
        plt.plot(
            nump(measurements_global[5:pcg]),
            numpr(noiselevel[3 : pcg - 2], 5),
            label="noise",
        )
        #
        plt.legend()
        plt.show()
        ###
        self.print_proof_records(P)
        return
