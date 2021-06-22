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

from historical import Historical
from utils import CoherenceError, itt, numpr


class Parameters:  # records various parameters
    def __init__(self, HST: Historical):
        #
        print("please enter alpha, beta and model_n")
        print(
            "alpha = | A - A^2 | and beta = | A^2 - A^3 |, we are doing |A^3 - A^4 | = | A^4 | = 1 and A is an associated-graded"
        )
        print(
            "ranges 2 <= alpha, beta <= 6 and alpha + beta <= 10, GPU needed for values bigger than around 3 or 4"
        )
        print("model_n governs the size of the neural networks")
        print(
            "suggested value n=4, can go to n=8 for more difficult cases on GPU"
        )
        alpha = int(input("input alpha : "))
        beta = int(input("input beta : "))
        model_n = int(input("input model_n : "))
        #
        #
        if alpha < 2 or beta < 2 or alpha > 6 or beta > 6 or alpha + beta > 10:
            print(
                "suggested range is alpha,beta in [2,...,6] and alpha + beta <= 10"
            )
            raise CoherenceError("exiting")
        if model_n < 1 or model_n > 12:
            print("suggested range for model_n is [2,...,12]")
            raise CoherenceError("exiting")
        #
        self.model_n = (
            model_n  # could go up to 10 (that was what we did before)
        )
        #
        self.alpha = alpha
        self.alpha2 = alpha * alpha
        self.alpha3 = alpha * alpha * alpha
        self.alpha3z = self.alpha3 + 1
        self.beta = beta
        self.betaz = self.beta + 1
        #
        HST.record_parameters(self.alpha, self.beta)
        #
        self.verbose = False
        #
        #
        self.rdetect_max = 1000
        #
        self.global_params = 0
        self.local_params = 0
        #
        # for rr1:
        #
        self.qvalue = 0.9
        #
        self.ascore_max = 7  # it is going to be a sum of a 0,1,2 and a 0,2,4
        #
        self.pastsize = 100
        self.futuresize = 1000
        #
        # for rr2:
        #
        self.profile_filter_on = True
        self.halfones_filter_on = True
        #
        # for rr3:
        self.chunksize = (
            1000  # reduced this: for the 5,5 case 1000 led to memory overflow
        )
        self.chunksize_extra = 10
        self.exponent = 0.9
        self.chunkconstant = 300
        self.chunkextent = 10
        #
        self.ukseuil = 0.5
        #
        self.newratio = 0.8  # how much we use the model rather than benchmark version to choose x,y
        self.search_epsilon = (
            0.3  # the proportion of randomized strategy choices
        )
        #
        # for rr4:
        self.prooflooplength = 4000
        self.done_max = 30000
        #
        # self.sleeptime = 120  # use this on a laptop
        self.sleeptime = 0  # was 5
        self.periodicity = 5
        self.stopthreshold = 100000  # too big for a notebook utilisation
        if torch.cuda.is_available():
            self.trainingiterations = 4  # was 8
        else:
            self.trainingiterations = 1
        #
        self.root_injection = 100
        self.randomize_q = 0.7
        self.randomize_factor = 0.1
        self.perturbation_factor = 0.3
        #
        self.spiral_mix_threshold = 10
        self.dropout_style = "regular"
        #
        #
        # info ranges   ###################
        #
        self.infosize = (
            300  # currently we just need up to 250 for sample info stuff
        )
        #
        self.samplepoints = 15
        self.sampleinfolower = 50
        self.sampleinfoupper = 250  # on ecrase vilower
        #
        self.fulldata_location = 25
        self.phase = 12
        #
        ###################################
        #
        self.basicloop_iterations = 3
        self.basicloop_training_iterations = 3
        #
        self.tweak_start = 200
        self.tweak_step = 4
        self.tweak_density = 0.05
        self.tweak_epsilon = 0.02
        self.tweak_decay = 0.9
        #
        self.splitting_probability = 1.0 - (
            10 ** (-2.0 / ((itt(self.alpha).to(torch.float)) ** 2))
        )
        print("splitting probability", numpr(self.splitting_probability, 4))
        #
        self.outlier_threshold = 2.5
        #
        ##### learner parameters
        #
        self.explore_max = 30000
        self.examples_max = 5000
        self.new_examples_max = 300
        self.new_explore_max = 400
        self.outlier_max = 2000
        self.new_outliers_max = 300
        #
        self.noise_level = 0.05
        self.noise_period = 50.0
        self.noise_decay = 0.5  # decay per period
        #
