#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Filename: /root/autodl-tmp/mol_prophecy/mytorchdrug/datasets/MySIDER.py
# Path: /root/autodl-tmp/mol_prophecy/mytorchdrug/datasets
# Created Date: Tuesday, June 11th 2024, 7:51:14 pm
# Author: Qiong Zhou
# 
# Copyright (c) 2024 Your Company
###
import os

from torchdrug import data, utils
from torchdrug.datasets import SIDER
from torchdrug.core import Registry as R
import json

@R.register("datasets.MySIDER")
class MySIDER(SIDER):
    """
    Marketed drugs and adverse drug reactions (ADR) dataset, grouped into 27 system organ classes.

    Statistics:
        - #Molecule: 1,427
        - #Classification task: 27

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/sider.csv.gz"
    md5 = "77c0ef421f7cc8ce963c5836c8761fd2"
    target_fields = None # pick all targets

    def __init__(self, path,chatgpt_oracle_path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.chatgpt_oracle_path = chatgpt_oracle_path

        # zip_file = utils.download(self.url, path, md5=self.md5)
        # csv_file = utils.extract(zip_file)

        csv_file = './data/sider.csv'
        self.load_csv(csv_file, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)
        self.load_map_txt(self.chatgpt_oracle_path)
    
    def load_map_txt(self,chatgpt_oracle_path):
        with open("{}/map.txt".format(chatgpt_oracle_path), 'r') as f:
            lines = f.readlines()
        self.map_dict = {}
        for line in lines:
            line = line.strip()
            if line:
                line_list = line.split('\t')
                with open(line_list[1], 'r') as file:
                    data = json.load(file)
                    self.map_dict[line_list[0]] = str(data["Insights"])

    def get_item(self, index):

        if self.smiles_list[index] in self.map_dict:
            chatgpt_json_file = self.map_dict[self.smiles_list[index]]
        else:
            chatgpt_json_file = self.smiles_list[index]

        if getattr(self, "lazy", False):
            # TODO: what if the smiles is invalid here?
            item = {
                "graph":
                data.Molecule.from_smiles(self.smiles_list[index],
                                          **self.kwargs),
                "smiles":self.smiles_list[index],
                "text":
                chatgpt_json_file
            }
        else:
            item = {"graph": self.data[index], "smiles":self.smiles_list[index], "text": chatgpt_json_file}
        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item
