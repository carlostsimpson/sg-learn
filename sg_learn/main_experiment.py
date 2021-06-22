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
"""
 to set up the environment: please run this cell and enter the desired alpha,beta and model_n
 (first suggested values are alpha = 3, beta = 2 and model_n = 4 )
"""
from driver import Driver
from historical import Historical
from parameters import Parameters
from proto_model import ProtoModel
from sg_model import SgModel

HST = Historical(10000)
# can be used to set the upper limit for proof lengths
HST.proof_nodes_max = 100000

# this will ask for alpha, beta and model_n
Pp = Parameters(HST)
Pp.basicloop_iterations = 3  # default is 3
Pp.basicloop_training_iterations = 3  # default is 3
Dd = Driver(Pp, HST)

# the following could be set to False to turn off those additional filters
Pp.profile_filter_on = True  # the default is True
Pp.halfones_filter_on = True  # the default is True

"""
please run this cell to (re)initialize the model
"""
Mm = SgModel(Pp)
Pp.spiral_mix_threshold = 4
Mmr = ProtoModel(Pp, "spiral_mix")

"""
do a classificationproof with Mmr to set benchmark, then one with Mm for the first data point
then do basicloop_classificationproof  for example for 50 iterations
the basicloop_classificationproof can be repeated. Cumulative results are printed in the output
(a first suggested value for (alpha,beta)=(3,2) could be sigma = 3)
"""

proving_instances, training_instances, title_text = Dd.instance_chooser()
HST.reset()
# sets the benchmark value into HST, using ProtoModel Mmr
Dd.classificationproof(Mmr, Mm, 0, proving_instances, title_text)
# # to record proof 0
Dd.classificationproof(Mm, Mm, 0, proving_instances, title_text)
Dd.basicloop_classificationproof(
    Mm, Mm, proving_instances, training_instances, title_text
)
