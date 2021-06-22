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
from minimizer import Minimizer
from relations_4 import Relations4
from utils import itp, nump


class MinimizerHistory:  # this becomes the first element of the relations datatype
    def __init__(self, model, HST: Historical):
        #
        #
        self.Mm = model
        self.Pp = self.Mm.pp
        self.Dd = Driver(self.Pp, HST)
        #
        sigma = int(input("input sigma : "))
        self.sigma = sigma
        #
        print("Minimizer History for sigma =", sigma)
        print(
            "with profile_filter_on =",
            self.Pp.profile_filter_on,
            "and halfones_filter_on =",
            self.Pp.halfones_filter_on,
        )

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
        instancevector, trainingvector, proof_title = self.Dd.InOne(self.sigma)
        self.InitialData = self.Dd.initialdata(instancevector, 0)
        #
        self.results = torch.zeros(
            (self.alpha, self.alpha, self.betaz), dtype=torch.int64, device=Dvc
        )
        #
        idlength = self.InitialData["length"]
        idprod = self.InitialData["prod"]
        self.availablexyp = self.rr1.availablexyp(idlength, idprod)[0]
        #

    def print_results(self):
        #
        results_sum = self.results.sum(2)
        #
        minimal = results_sum[0, 0]
        for x in range(self.alpha):
            for y in range(self.alpha):
                if results_sum[x, y] < minimal:
                    minimal = results_sum[x, y]
        root_minimum = minimal + 1
        print(
            "=+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+="
        )
        print(
            "=+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+="
        )
        print(
            "               aggregation of results for sigma =",
            itp(self.sigma),
        )
        print(
            "=+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+="
        )
        print(
            "=+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+=   =+="
        )
        #
        print("summed results according to initial cut location are:")
        print(nump(results_sum))
        print(
            "the minimal number of nodes including the root is",
            itp(root_minimum),
        )
        print(
            "this was with profile_filter_on =",
            self.Pp.profile_filter_on,
            "and halfones_filter_on =",
            self.Pp.halfones_filter_on,
        )
        print(
            "---------------------------------------------------------------------"
        )
        return

    def minimize_all(self):
        #
        for x in range(self.alpha):
            for y in range(self.alpha):
                for p in range(self.betaz):
                    if self.availablexyp[x, y, p]:
                        Min = Minimizer(self.Mm, self.sigma, x, y, p)
                        cut_instance = Min.down[0, x, y, p]
                        assert (
                            Min.lowerbound[cut_instance]
                            == Min.upperbound[cut_instance]
                        )
                        self.results[x, y, p] = Min.lowerbound[cut_instance]
        self.print_results()
        return
