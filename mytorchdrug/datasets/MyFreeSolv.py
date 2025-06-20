import os

from torchdrug import data, utils
from torchdrug.datasets import FreeSolv
from torchdrug.core import Registry as R
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

@R.register("datasets.MyFreeSolv")
class MyFreeSolv(FreeSolv):
    """
    Experimental and calculated hydration free energy of small molecules in water.

    Statistics:
        - #Molecule: 642
        - #Regression task: 1

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/FreeSolv.zip"
    md5 = "8d681babd239b15e2f8b2d29f025577a"
    target_fields = ["expt"]

    def __init__(self, path,chatgpt_oracle_path, verbose=1, **kwargs):
        super().__init__(path, verbose=verbose, **kwargs)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.chatgpt_oracle_path = chatgpt_oracle_path

        zip_file = utils.download(self.url, self.path, md5=self.md5)
        csv_file = utils.extract(zip_file, "SAMPL.csv")

        self.load_csv(csv_file,
                      smiles_field="smiles",
                      target_fields=self.target_fields,
                      verbose=verbose,
                      **kwargs)
        
        self.load_map_txt(self.chatgpt_oracle_path)

        self.remove_outliers()
        
        # self.normalize_targets()
        # self.load_error_txt(self.chatgpt_oracle_path)
    
    def remove_outliers(self):
        targets = np.array(self.targets['expt'])
        Q1 = np.percentile(targets, 25)
        Q3 = np.percentile(targets, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        mask = (targets >= lower_bound) & (targets <= upper_bound)

        self.smiles_list = [s for s, m in zip(self.smiles_list, mask) if m]
        self.targets['expt'] = [t for t, m in zip(self.targets['expt'], mask) if m]
        self.data = [s for s, m in zip(self.data, mask) if m]
        self.map_dict = {k:v for (k,v),m in zip(self.map_dict.items(),mask) if m}
    
    def normalize_targets(self):
        self.scaler = MinMaxScaler()
        self.original_targets = np.array(self.targets['expt'])
        self.targets['expt'] = self.scaler.fit_transform(self.original_targets.reshape(-1, 1)).flatten()
    
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

    def load_error_txt(self,chatgpt_oracle_path):
        with open("{}/error.txt".format(chatgpt_oracle_path), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if './data/freesolv/oracles' in line:
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
