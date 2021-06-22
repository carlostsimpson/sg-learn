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
#### do the proofs in a range

for i in range(13):
    print("instance", i)
    proving_instances, training_instances, title_text = Dd.InOne(i)
    Dd.classificationproof(Mm, Mm, 0, proving_instances, title_text)

#######################################################################################


Pp.profile_filter_on = True
Pp.halfones_filter_on = True

MH = MinimizerHistory(Mmr)  # asks to choose sigma
MH.minimize_all()  # this does all the cases in a row

Min = Minimizer(Mm, 3, 0, 0, 0)  # individual cases: sigma, x, y, p
Min.check_done_print()
# time.sleep(60)

Min = Minimizer(Mm, 3, 0, 0, 1)  # individual cases: sigma, x, y, p
Min.check_done_print()
# time.sleep(60)

Min = Minimizer(Mm, 3, 0, 0, 2)  # individual cases: sigma, x, y, p
Min.check_done_print()
# time.sleep(60)


Fws = FindWeirdStuff(Dd, Mm)

### a few things to do with that

Fws.tracer_root(5)

Fws.tracer_subroot(5, 0, 0, 0)

Fws.show_cut_column(5, 2, 1)

Fws.print_one_sample_from_box(0.25, 0.35, 0.0, 0.2)

Fws.print_one_sample_from_box(0.0, 0.05, 0.0, 0.2)
