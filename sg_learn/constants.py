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

if torch.cuda.is_available():
    Dvc = torch.device("cuda:0")
    print("Running on GPU cuda:0")
else:
    Dvc = torch.device("cpu")
    print("Running on CPU")

CpuDvc = torch.device("cpu")


torch_pi = torch.acos(torch.zeros(1)).item() * 2
