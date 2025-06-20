import os
from omegaconf import OmegaConf
from mol_prophecy.model.mols.graph_expert import GraphExpert
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))


def test_graph_model_loading():
    cfg = OmegaConf.load("../config/config.yaml")
    graph_model = GraphExpert(
        pretrained_model_path=os.path.join(os.getenv("PRETRAINED_ROOT"),
                                           cfg.Expert.graph.model,
                                           "model.pth"),
        feat_dim=cfg.Expert.graph.feat_dim,
        emb_dim=cfg.Expert.graph.emb_dim,
        edge_feature_dim=None,
        num_graph_layers=cfg.Expert.graph.num_graph_layers,
        drop_ratio=cfg.Expert.graph.drop_ratio,
        batch_norm=True,
        readout="mean")

    assert graph_model is not None


def test_graph_single_encoding():
    from torchdrug import data

    smi = "C1=CC=CC=C1"
    # mol = data.Molecule.from_smiles("C1=CC=CC=C1")
    # print(mol.node_feature.shape[1])
    # print(mol.edge_feature.shape[1])

    cfg = OmegaConf.load("../config/config.yaml")

    graph_model = GraphExpert(
        pretrained_model_path=os.path.join(os.getenv("PRETRAINED_ROOT"),
                                           cfg.Expert.graph.model,
                                           "model.pth"),
        feat_dim=cfg.Expert.graph.feat_dim,
        emb_dim=cfg.Expert.graph.emb_dim,
        edge_feature_dim=None,
        num_graph_layers=cfg.Expert.graph.num_graph_layers,
        drop_ratio=cfg.Expert.graph.drop_ratio,
        batch_norm=True,
        readout="mean")

    assert graph_model is not None

    encoding = graph_model([smi] * 10)["graph_feature"]

    # assert "graph_feature" in encoding
    # assert "node_feature" in encoding
    # assert encoding["node_feature"].shape[0] == mol.num_atom

    assert encoding.shape[0] == 10
    assert encoding.shape[1] == cfg.Expert.graph.feat_dim
