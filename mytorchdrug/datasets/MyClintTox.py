import os
import json
from torchdrug import data, utils
from torchdrug.datasets import ClinTox
from torchdrug.core import Registry as R


@R.register("datasets.MyClinTox")
class MyClinTox(ClinTox):
    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
    md5 = "db4f2df08be8ae92814e9d6a2d015284"
    target_fields = ["FDA_APPROVED", "CT_TOX"]

    def __init__(self, path,chatgpt_oracle_path, verbose=1, **kwargs):
        super().__init__(path, verbose=verbose, **kwargs)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.chatgpt_oracle_path = chatgpt_oracle_path

        # zip_file = utils.download(self.url, path, md5=self.md5)
        # csv_file = utils.extract(zip_file)
        csv_file = './data/clintox.csv'

        self.load_csv(csv_file,
                      smiles_field="smiles",
                      target_fields=self.target_fields,
                      verbose=verbose,
                      **kwargs)

        self.load_map_txt(self.chatgpt_oracle_path)

        # self.load_error_txt(self.chatgpt_oracle_path)

    def load_map_txt(self,chatgpt_oracle_path):
        with open("{}/map.txt".format(chatgpt_oracle_path), 'r') as f:
            lines = f.readlines()
        self.map_dict = {}
        for line in lines:
            line = line.strip()
            if line:
                line_list = line.split('\t')
                if line_list[0] == '*C(=O)[C@H](CCCCNC(=O)OCCOC)NC(=O)OCCOC':
                    continue
                with open(line_list[1], 'r') as file:
                    data = json.load(file)
                    self.map_dict[line_list[0]] = str(data["Insights"])


    def load_error_txt(self,chatgpt_oracle_path):
        with open("{}/error.txt".format(chatgpt_oracle_path), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if './data/clintox/oracles' in line:
                line_list = line.split('\t')
                with open(line_list[1], 'r') as file:
                    data = json.load(file)
                    self.map_dict[line_list[0]] = str(data["Insights"])

    def get_item(self, index):
        
        if self.smiles_list[index] in self.map_dict:
            # get text json file
            chatgpt_json_file = self.map_dict[self.smiles_list[index]]
        else:
            chatgpt_json_file = ''

        if getattr(self, "lazy", False):
            # TODO: chage smiles to real text from chatgpt
            item = {
                "graph":
                data.Molecule.from_smiles(self.smiles_list[index],
                                          **self.kwargs),
                "smiles":self.smiles_list[index],
                "text":
                chatgpt_json_file
            }
        else:
            # TODO: chage smiles to real text from chatgpt
            
            item = {"graph": self.data[index],"smiles":self.smiles_list[index], "text": chatgpt_json_file}

        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item
