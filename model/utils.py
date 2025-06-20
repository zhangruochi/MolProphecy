#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/mol_prophecy/model/utils.py
# Project: /home/richard/projects/mol_prophecy/model
# Created Date: Saturday, April 27th 2024, 3:23:26 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Sat Apr 27 2024
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2024 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2024 Ruochi Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###

# ['MW', 'ALOGP', 'HBA', 'HBD', 'PSA', 'ROTB', 'AROM', 'ALERTS']
# use rdkit to calculate the physicochemical properties


def cal_physicochemical_properties(smi, all_features):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import AllChem
    import numpy as np
    import pandas as pd

    mol = Chem.MolFromSmiles(smi)
    res = []
    if "MW" in all_features:
        MW = Descriptors.MolWt(mol)
        res.append(MW)
    if "ALOGP" in all_features:
        ALOGP = Descriptors.MolLogP(mol)
        res.append(ALOGP)
    if "HBA" in all_features:
        HBA = Descriptors.NumHAcceptors(mol)
        res.append(HBA)
    if "HBD" in all_features:
        HBD = Descriptors.NumHDonors(mol)
        res.append(HBD)
    if "PSA" in all_features:
        PSA = Descriptors.TPSA(mol)
        res.append(PSA)
    if "ROTB" in all_features:
        ROTB = Descriptors.NumRotatableBonds(mol)
        res.append(ROTB)
    if "AROM" in all_features:
        AROM = Descriptors.NumAromaticRings(mol)
        res.append(AROM)

    return np.array(res)
