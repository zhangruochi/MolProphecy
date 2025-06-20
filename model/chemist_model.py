#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/mol_prophecy/model/chemist_model.py
# Project: /home/richard/projects/mol_prophecy/model
# Created Date: Wednesday, January 31st 2024, 11:41:27 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Fri Apr 26 2024
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
from openai import OpenAI
from dotenv import load_dotenv
from ..chemist.utils import get_response_from_chat, calculate_properties
import json


class ChemistPredictor():

    def __init__(self, cfg):
        load_dotenv(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                ".env"))
        self.cfg = cfg
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), )
        self.PROMPT_PATH = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            self.cfg.chemist.predictor.prompt_path,
            "{}.txt".format(self.cfg.chemist.predictor.version))
        self.TASK = self.cfg.task_desc[self.cfg.train.dataset]

    def __call__(self, smiles):
        with open(self.PROMPT_PATH, "r") as f:
            prompt_temp = f.read()
        # use rdkit to calculate the properties MW, logP, HBD, HBA, RB, TPSA
        properties = calculate_properties(smiles)
        gpt_input = prompt_temp.format(SMILES=smiles,
                                       TASK=self.TASK,
                                       MW=properties["MW"],
                                       CLogP=properties["CLogP"],
                                       HBD=properties["HBD"],
                                       HBA=properties["HBA"],
                                       RB=properties["RB"],
                                       TPSA=properties["TPSA"])

        str_ = get_response_from_chat(self.client, gpt_input)

        try:
            prediction = json.loads(str_)
            prediction["status"] = "success"
        except:
            prediction["status"] = "fail"

        return prediction
