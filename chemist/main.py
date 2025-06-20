#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/de_composer/chat/main.py
# Project: /home/richard/projects/mol_prophecy/chemist
# Created Date: Tuesday, August 15th 2023, 5:42:43 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Fri May 24 2024
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2023 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2023 Ruochi Zhang
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
from tqdm import tqdm
from .utils import get_oracle_from_chemist
import openai
from openai import OpenAI
from dotenv import load_dotenv


class Chemist(object):

    def __init__(self, cfg):

        load_dotenv(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

        self.cfg = cfg

        self.PROMPT_PATH = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            self.cfg.chemist.prompt.path,
            "{}.txt".format(self.cfg.chemist.prompt.version))

        self.TASK = self.cfg.task_desc[self.cfg.train.dataset]
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"))

    def talk(self, SMILES: str):

        with open(self.PROMPT_PATH, "r") as f:
            prompt_temp = f.read()

        return get_oracle_from_chemist(self.client, prompt_temp, SMILES,
                                       self.TASK)
