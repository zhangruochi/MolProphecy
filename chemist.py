#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/mol_prophecy/get_chemist_instruction.py
# Project: /home/richard/projects/mol_prophecy
# Created Date: Thursday, January 18th 2024, 11:43:26 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Wed Jan 31 2024
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
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from omegaconf import OmegaConf

from chemist.main import Chemist
import json

if __name__ == "__main__":

    df = pd.read_csv("./data/DrugBankApproved.csv")
    smiles = df.smiles.tolist()[0]
    cfg = OmegaConf.load("./config/config.yaml")

    chemist = Chemist(cfg)
    result = chemist.talk(smiles)

    with open("test.json", "w") as f:
        json.dump(json.loads(result), f)
