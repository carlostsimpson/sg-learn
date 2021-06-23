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
import gc

import numpy as np
import torch

from constants import CpuDvc, Dvc


class CoherenceError(Exception):
    pass


def zbinary(
    depth, z
):  # the smallest digits are first here (i.e. reads backward)
    binnum = [int(i) for i in bin(z)[2:]]
    bnlength = len(binnum)
    if bnlength > depth:
        print(
            "warning: applying zbinary to",
            z,
            "with depth",
            depth,
            "but bin length was",
            bnlength,
        )
    outputarray = torch.zeros((depth), dtype=torch.bool, device=Dvc)
    for j in range(depth):
        if j < bnlength:
            outputarray[j] = binnum[bnlength - j - 1] == 1
        else:
            outputarray[j] = False
    return outputarray


def binaryzbatch(length, depth, binarray_batch):
    zbatch = torch.zeros((length), dtype=torch.int64, device=Dvc)
    for i in range(depth):
        zbatch += binarray_batch.to(torch.int)[:, i] * (2 ** i)
    return zbatch


def composedetections(
    length, detection1, detection2
):  # outputs the result of detection vector 2 inserted in detection1
    output = torch.zeros((length), dtype=torch.bool, device=Dvc)
    output[detection1] = detection2
    return output


def memReport(style):  # by QuantScientist Solomon K @smth
    if style == "memory":
        print("gc memory report")
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                print(type(obj), obj.size())
    if style == "mg":
        print("gc memory and garbage report::", end=" ")
        allobjects = gc.get_objects()
        all_length = len(allobjects)
        elements = torch.zeros((all_length), dtype=torch.int64, device=Dvc)
        count = 0
        loc = 0
        for obj in allobjects:
            if torch.is_tensor(obj):
                count += 1
                elements[loc] = torch.numel(obj)
            loc += 1
        elcount = elements.sum(0)
        print(
            "there are",
            count,
            "torch tensors in play with",
            itp(elcount),
            "elements",
        )
        #
        _, indices = torch.sort(elements, descending=True)
        upper = 5
        if upper > count:
            upper = count
        for i in range(upper):
            indi = indices[i]
            obji = allobjects[indi]
            print(obji.size())
        print("|||")
        for obj in gc.garbage:
            if torch.is_tensor(obj):
                print(type(obj), obj.size())


def arangeic(x):
    ar = torch.arange(x, dtype=torch.int64, device=Dvc)
    return ar


def itp(x):  # integer to print
    return nump(x)


def itt(x):  # integer to torch
    if torch.is_tensor(x):
        return x
    return torch.tensor(x, device=Dvc)


def itf(x):  # integer to torch.float
    return itt(x).to(torch.float)


def nump(x):
    if torch.is_tensor(x):
        return x.detach().to(CpuDvc).numpy()
    return x


def numpr(x, k):
    return np.round(nump(x), k)


def numpi(x):
    return nump(x.to(torch.int))
