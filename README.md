# MolProphecy: Multimodal Molecular Property Prediction

MolProphecy is a cutting-edge multimodal molecular property prediction system that combines molecular graph encoding with chemical expert insights from ChatGPT. This system leverages both structural molecular information and expert knowledge to achieve superior prediction performance across various molecular property prediction tasks.

![](./docs/MolProphecy.png)

## 🚀 Features

- **Multimodal Fusion**: Combines molecular graph representations with expert text analysis
- **Multiple LLM Support**: Supports both BERT and LLaMA models for text encoding
- **Flexible Molecular Encoding**: Supports both graph-based (MolCLR) and sequence-based (ChemBERTa) molecular representations
- **Comprehensive Dataset Support**: Pre-configured for multiple molecular property prediction datasets
- **Advanced Fusion Methods**: Multiple fusion approaches including attention, tensor fusion, and bilinear fusion
- **Handcrafted Features**: Optional integration of traditional molecular descriptors
- **GPU Memory Optimization**: Cache support for large language models

## 📋 Supported Tasks

- **BACE**: Binary binding prediction for BACE-1 inhibitors
- **ClinTox**: Clinical toxicity prediction
- **FreeSolv**: Hydration free energy prediction
- **SIDER**: Side effect prediction (27 categories)

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- Conda or Miniconda

### Environment Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd MolProphecy
```

2. **Create and activate conda environment**:
```bash
conda env create -f environment.yaml
conda activate mol
```

3. **Install PyTorch Geometric** (if not already included):
```bash
pip install torch-geometric torch-scatter torch-cluster
```

4. **Set up environment variables**:
Create a `.env` file in the project root:
```bash
PRETRAINED_ROOT=/path/to/pretrained/models
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1  # Optional, for custom endpoints
```

## 📥 Download Pretrained Models

### Language Models (HuggingFace)

Download required language models from HuggingFace:

```bash
cd scripts
python download_huggingface.py --model bert-base-cased
python download_huggingface.py --model "meta-llama/Meta-Llama-3-8B"
```

### MolCLR Pre-trained Models

Download MolCLR pre-trained weights from the [official MolCLR repository](https://github.com/yuyangw/MolCLR):

```bash
# Clone the MolCLR repository
git clone https://github.com/yuyangw/MolCLR.git
cd MolCLR

# Download pre-trained models
# The pre-trained models are available in:
# - ckpt/pretrained_gin/ (GIN model)
# - ckpt/pretrained_gcn/ (GCN model)

# Copy the model files to your MolProphecy project
cp -r ckpt/pretrained_gin /path/to/MolProphecy/molclr/
cp -r ckpt/pretrained_gcn /path/to/MolProphecy/molclr/
```

### Available Models:
- **Language Models**:
  - `bert-base-cased`
  - `meta-llama/Meta-Llama-3-8B`
- **Molecular Models**:
  - `molclr` (molecular graph model) - [MolCLR Official Repository](https://github.com/yuyangw/MolCLR)
  - `ChemBERTa_zinc250k_v2_40k` (molecular sequence model)

### Model Setup

After downloading, ensure your directory structure looks like:
```
MolProphecy/
├── molclr/
│   ├── pretrained_gin/
│   │   └── model.pth
│   └── pretrained_gcn/
│       └── model.pth
└── ...
```

## ⚙️ Configuration

The main configuration file is `config/config.yaml`. Key settings include:

### Model Configuration
```yaml
LLMs:
  model: llama  # bert or llama
  freeze: True
  llama:
    use_cache: True  # Enable for GPU memory optimization

Expert:
  model: graph  # graph or sequence
  freeze: False
  graph:
    model: "./molclr"
    feat_dim: 512
  sequence:
    model: "./ChemBERTa_zinc250k_v2_40k"
```

### Training Configuration
```yaml
train:
  dataset: bace  # bace, sider, freesolv, clintox, etc.
  data_split: "scaffold"  # scaffold or random
  learning_rate: 3e-4
  batch_size: 128
  num_epoch: 150
  fusion_approach: "attention"  # concat, tensor_fusion, bilinear_fusion, attention
```

## 🚀 Usage

### Training

1. **Generate LLM cache** (for GPU memory optimization):
```bash
python llama3_gen.py
```

2. **Start training**:
```bash
python fusion_train.py
```

### Inference

```bash
python inference.py
```

### Evaluation

```bash
python evaluate.py
```

## 🧪 Unit Testing

Run all unit tests:
```bash
cd unit_test
pytest
```

Run specific test modules:
```bash
pytest test_bert_model.py
pytest test_graph_model.py
pytest test_prophecy.py
```

## 📊 Model Architecture

### Core Components

1. **Molecular Expert Module**:
   - Graph-based: MolCLR with GNN layers
   - Sequence-based: ChemBERTa for SMILES encoding

2. **Language Model Module**:
   - BERT: For general text encoding
   - LLaMA: For advanced language understanding

3. **Fusion Module**:
   - Attention-based fusion with GatedXattnDenseLayer
   - Multiple fusion strategies (concat, tensor_fusion, bilinear_fusion)
   - Handcrafted feature integration

4. **Prediction Head**:
   - Multi-layer perceptron with configurable architecture
   - Support for both classification and regression tasks

### Data Flow

```
SMILES → Molecular Expert → Molecular Embeddings
    ↓
Task Description → LLM → Text Embeddings
    ↓
Fusion Module → Combined Representations
    ↓
Prediction Head → Property Prediction
```

## 📈 Performance

The system achieves state-of-the-art performance across multiple molecular property prediction benchmarks:

- **BACE**: Binary classification for BACE-1 inhibitors
- **SIDER**: Multi-label classification for 27 side effects
- **FreeSolv**: Regression for hydration free energy
- **ClinTox**: Binary classification for clinical toxicity

## 🔧 Advanced Features

### Ablation Studies

Enable ablation experiments in `config.yaml`:
```yaml
train:
  Ablation:
    is_ablation: True
    experiment_model: graph  # bert, llama, graph, sequence
```

### Handcrafted Features

Integrate traditional molecular descriptors:
```yaml
Handcrafted:
  use_handcrafted_features: True
  feature_dim: 200
```

### NNI Integration

Enable Neural Network Intelligence for hyperparameter optimization:
```yaml
mode:
  nni: True
```

## 📁 Project Structure

```
MolProphecy/
├── config/                 # Configuration files
├── data/                   # Dataset files
├── model/                  # Model implementations
│   ├── llms/              # Language model modules
│   ├── mols/              # Molecular encoding modules
│   └── seqs/              # Sequence-based modules
├── chemist/               # ChatGPT integration
├── scripts/               # Utility scripts
├── unit_test/             # Test files
├── result/                # Experimental results
└── utils/                 # Utility functions
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


## 📞 Contact

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note**: Make sure to set the correct configuration in `config.yaml` before using the model:
- `classify_task`: True for classification tasks
- `is_ablation`: False for normal training
- `Expert.model`: Choose between 'sequence' or 'graph'
