#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/mol_prophecy/test.py
# Project: /home/richard/projects/mol_prophecy/unit_test
# Created Date: Saturday, April 27th 2024, 12:45:48 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Thu Jun 13 2024
# Modified By: Qiong Zhou
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
import json
import os
from tqdm import tqdm


def test_bace():
    # list all json file
    dir = "../data/bace/oracles"
    files = os.listdir(dir)

    print("\nnumber of files in bace: ", len(files))

    for file in tqdm(files, desc="Checking BACE", total=len(files)):
        with open(os.path.join(dir, file)) as f:
            data = json.load(f)
            assert data is not None
            assert "Insights" in data
            assert "Query SMILES" in data
            assert len(data["Insights"]) > 0


def test_freesolv():
    # list all json file
    dir = "../data/freesolv/oracles"
    files = os.listdir(dir)

    print("\nnumber of files in freesolv: ", len(files))

    for file in tqdm(files, desc="Checking FreeSolv", total=len(files)):
        with open(os.path.join(dir, file)) as f:
            data = json.load(f)
            assert data is not None
            assert "Insights" in data
            assert "Query SMILES" in data
            assert len(data["Insights"]) > 0


def test_sider():
    # list all json file
    dir = "../data/sider/oracles"
    files = os.listdir(dir)

    print("\nnumber of files in sider: ", len(files))

    for file in tqdm(files, desc="Checking Sider", total=len(files)):
        with open(os.path.join(dir, file)) as f:
            data = json.load(f)
            assert data is not None
            assert "Insights" in data
            assert "Query SMILES" in data
            assert len(data["Insights"]) > 0
