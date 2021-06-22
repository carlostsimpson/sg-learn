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
this is the end of the main notebook part, further optional cells are included below
also note that the previous basicloop_classificationproof cell can be repeated, cumulating the proof history
"""

# a function to view the initial data with instance input as above
Dd.print_instances()

# InList use case (this isn't covered by the instance chooser)
# modify the following as desired: proving list, training list
proving_instances, training_instances, title_text = Dd.InList(
    [6, 7], [5, 8, 9, 10, 11]
)
HST.reset()
Dd.classificationproof(Mmr, Mm, 0, proving_instances, title_text)
Dd.classificationproof(Mm, Mm, 0, proving_instances, title_text)
Dd.basicloop_classificationproof(
    Mm, Mm, proving_instances, training_instances, title_text
)
