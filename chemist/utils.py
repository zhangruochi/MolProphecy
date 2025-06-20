import os
import openai
import re
import pandas as pd
from typing import List
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski


def calculate_properties(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {
            "MW": "undefined",
            "CLogP": "undefined",
            "HBD": "undefined",
            "HBA": "undefined",
            "RB": "undefined",
            "TPSA": "undefined"
        }

    properties = {
        "MW": round(Descriptors.MolWt(mol), 3),
        "CLogP": round(Descriptors.MolLogP(mol), 3),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RB": Lipinski.NumRotatableBonds(mol),
        "TPSA": round(Descriptors.TPSA(mol), 3)
    }

    return properties


def get_response_from_chat(client, prompt: str) -> str:
    """ Get response from GPT chatbot
    """

    # model = os.getenv("model")
    model = "gpt-4o"
    model = "deepseek-chat"
    # print(prompt)
    # print(model)
    response = client.chat.completions.create(model=model,
                                              messages=[{
                                                  "role": "user",
                                                  "content": prompt
                                              }])
    message = response.choices[0].message.content

    # remove ```json ``` in the response
    if "```json" in message:
        message = re.sub(r"```json\n", "", message)
    if "```" in message:
        message = re.sub(r"```", "", message)
    return message


def get_oracle_from_chemist(client, prompt: str, SMILES: str, TASK: str):
    """ 
    """
    properties = calculate_properties(SMILES)
    gpt_input = prompt.format(SMILES=SMILES,
                              TASK=TASK,
                              MW=properties["MW"],
                              CLogP=properties["CLogP"],
                              HBD=properties["HBD"],
                              HBA=properties["HBA"],
                              RB=properties["RB"],
                              TPSA=properties["TPSA"])

    str_ = get_response_from_chat(client, gpt_input)
    # print("raw result: \n" + str_)
    return str_
