import os

from torchdrug import data, utils
from torchdrug.datasets import BACE
from torchdrug.core import Registry as R
import json


@R.register("datasets.MyBACE")
class MyBACE(BACE):
    r"""
    Binary binding results for a set of inhibitors of human :math:`\beta`-secretase 1(BACE-1).

    Statistics:
        - #Molecule: 1,513
        - #Classification task: 1

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/bace.csv"
    md5 = "ba7f8fa3fdf463a811fa7edea8c982c2"
    target_fields = ["Class"]

    def __init__(self, path, chatgpt_oracle_path, verbose=1, **kwargs):
        super().__init__(path, verbose=verbose, **kwargs)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.chatgpt_oracle_path = chatgpt_oracle_path

        file_name = utils.download(self.url, path, md5=self.md5)

        self.load_csv(file_name,
                      smiles_field="mol",
                      target_fields=self.target_fields,
                      verbose=verbose,
                      **kwargs)
        self.load_map_txt(self.chatgpt_oracle_path)

    def load_map_txt(self, chatgpt_oracle_path):
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

        smi = self.smiles_list[index]
        chatgpt_json_file = self.map_dict[smi]

        if getattr(self, "lazy", False):
            item = {
                "smiles":
                self.smiles_list[index],
                "text":
                self.map_dict[self.smiles_list[index]],
                "graph":
                data.Molecule.from_smiles(self.smiles_list[index],
                                          **self.kwargs),
            }

        else:
            item = {
                "smiles":
                self.smiles_list[index],
                "text":
                self.map_dict[self.smiles_list[index]],
                "graph":
                data.Molecule.from_smiles(self.smiles_list[index],
                                          **self.kwargs),
            }

        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item
