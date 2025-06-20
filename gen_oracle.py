#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/mol_prophecy/data/generate_des.py
# Project: /home/richard/projects/mol_prophecy
# Created Date: Sunday, January 28th 2024, 1:30:31 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue Jun 11 2024
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

import sys
import os
import json
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from mol_prophecy.chemist.main import Chemist
from tqdm import tqdm

dataset_name = "sider"

if not os.path.exists("./data/{}".format(dataset_name)):
    os.makedirs("./data/{}".format(dataset_name), exist_ok=True)

df = pd.read_csv("./data/{}.csv".format(dataset_name))
smiles_list = df.smiles.tolist()
print("total smiles: ", len(smiles_list))
cfg = OmegaConf.load("./config/config.yaml")
chemist = Chemist(cfg)

with open("./data/{}/error.txt".format(dataset_name), "a") as error_f:
    with open("./data/{}/map.txt".format(dataset_name), "a") as map_f:

        os.makedirs("./data/{}/oracles".format(dataset_name), exist_ok=True)

        for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list)):
            try:
                result = chemist.talk(smiles)
                json_file_name = "./data/{}/oracles/{}.json".format(
                    dataset_name, i)
                with open(json_file_name, "w") as json_f:
                    json.dump(json.loads(result), json_f)
                map_f.write("{}\t{}\n".format(smiles, json_file_name))
            except Exception as e:
                print(e)
                error_f.write("{}\t{}\t{}\n".format(smiles, json_file_name,
                                                    result))
