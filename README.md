# LOKI: Latent-Space Optimization for Knowledge Integration

## Codebase for the VLDB 2027 Paper: <br/>**"Discovery-Driven Integration of Disjoint Tables via Text"**

---

## Overview

Relational databases in complex domains (such as healthcare and enterprise data lakes) frequently lack explicit foreign keys and shared primary schemas. Traditional schema matching and entity resolution techniques fail when tables share neither common attribute names nor overlapping key distributions.

**LOKI** (**L**atent-space **O**ptimization for **K**nowledge **I**ntegration) discovers and materializes cross-table joins by leveraging unstructured narrative text (e.g., clinical notes, protocol documents, progress summaries) as text mediated join-paths. Rather than relying on static schema alignment or monolithic prompt-based LLM generation, LOKI:

1. **Contextualizes Table Rows and Sentences**: Maps relational rows and unstructured text into a shared latent space using bidirectional cross-attention with query-dependent gating.
2. **Discovers Candidate Join Paths**: Scores row–sentence pairs to isolate high-confidence semantic links across disjoint tables.
3. **Clusters and Materializes Relational Tables**: Groups topological paths into high-purity relationship clusters and materializes relational tables for distinct semantic relationship types.

---

## Repository Structure

```
.
├── Datasets/
│   ├── Datasets/                  # Benchmark datasets (MIMIC-IV, Pharma, Feverous, Protrix)
│   └── MIMIC_Annotation_Pipeline/ # Extraction and clinical annotation pipelines for MIMIC-IV
│
├── LOKI/                          # Core model architecture
│   ├── run_cross_attention.py     # Main bidirectional cross-attention training driver
│   ├── bidirectional_cross_attention.py # Cross-attention layers and gating modules
│   ├── losses.py                  # 5-term loss formulation (InfoNCE, SIGReg, Sinkhorn, etc.)
│   └── README.md                  # Detailed architecture specs and Mermaid diagrams
│
├── Experiment-1/                  # Post-training ablation studies
│   ├── Post_Training_Evals/       # 5-model evaluation (Baseline, FT-Encoder, Uni R->S, Uni S->R, LOKI)
│   └── README.md                  # Reproduction instructions and checkpoint downloads
│
├── Experiment-2/                  # SOTA data discovery benchmarks
│   ├── SOTA_Evaluation_New/       # Comparative evaluation suite (CMDL, TaBERT, TabSTAR, LOKI)
│   ├── run_comparison_pharma.py   # Evaluation driver on Pharma Protocol
│   ├── run_scalability_pharma.py  # Search-space candidate pool scalability study
│   └── README.md                  # Setup, dependencies, and evaluation guide
│
├── Experiment-3/                  # End-to-end table materialization & LLM evaluation
│   ├── LLM_Eval_Ex-3/             # Pipeline execution code, GT annotations, and inference outputs
│   ├── #Results/                  # Tabular metrics, contingency matrices, and publication figures
│   └── README.md                  # Comprehensive benchmark results across 382 MIMIC-IV admissions
│
├── requirements.txt               # Pinned runtime environment dependencies
└── LICENSE.txt                    # GNU General Public License v3.0
```

---

## Technical Highlights

### 1. Bidirectional Cross-Attention
LOKI projects linearized table rows and document sentences into a shared latent space via a dual-path architecture:
- **Forward Path (Rows $\to$ Sentences)**: Linearized row queries attend to sentence key/value representations.
- **Reverse Path (Sentences $\to$ Rows)**: Sentence queries attend to row key/value representations.
- Both directions incorporate an inner sparse-attention gate and an outer query-dependent vector gate (`AttentionOutputGate`) before residual summation.

### 2. Multi-Objective Training Loss
Training uses a five-term objective designed to enforce alignment while preventing dimensional collapse:
- **Triplet / Ranking Loss (InfoNCE)**: Separates positive global similarity from hard negative pairings.
- **Pair Contrastive Loss**: Margin-based separation directly on the $N \times M$ row–sentence pair score matrix.
- **Attention Distillation Loss**: Jensen–Shannon divergence aligning student pair scores to a hub-centered frozen-encoder teacher.
- **Epps–Pulley SIGReg**: Penalizes non-Gaussianity on post-attention contextualized embeddings to prevent latent collapse.
- **Sinkhorn Marginal Regularization**: Enforces balanced marginal distributions across attention weights to suppress universal hub tokens.

### 3. Relationship-Type Table Materialization
Rather than producing unverified single-table dumps, LOKI discovers and materializes fine-grained relation tables corresponding to distinct cross-table relationship types. On 382 MIMIC-IV admissions:
- **Pair Precision**: $97.95\%$ (GPT-OSS 20B) / $98.00\%$ (Qwen-3.6).
- **Physical Table Micro Precision**: $84.0\%$ / $84.8\%$.
- **Cluster Structural Purity**: $\ge 99.5\%$ with Adjusted Rand Index (ARI) of $0.806$–$0.858$.

---

## Environment Setup

### Prerequisites
- Python 3.12+
- PyTorch 2.9+ with CUDA 12.8 support
- GPU with $\ge 16$ GB VRAM (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/dtim-upc/LOKI.git
cd LOKI

# Create and activate a virtual or conda environment
conda create -n loki python=3.12 -y
conda activate loki

# Install PyTorch with CUDA 12.8 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

### Pretrained Checkpoints
Published evaluation models and assets are available on Hugging Face:
- Model repository: [`shaoncsecu/LOKI`](https://huggingface.co/shaoncsecu/LOKI)
- Automated download scripts are provided within each experiment directory:
  - Experiment 1: `python Experiment-1/Post_Training_Evals/model_download.py`
  - Experiment 2: `python Experiment-2/SOTA_Evaluation_New/download_models.py`

---

## Experimental Reproduction

### Experiment 1: Architectural Ablations & Post-Training Analysis
Evaluates the contribution of bidirectional attention and loss terms across five model configurations: `Baseline` (frozen encoder), `FT-Encoder`, `Uni (R->S)`, `Uni (S->R)`, and `LOKI`.
- Guide: [`Experiment-1/README.md`](Experiment-1/README.md)
- Primary script: `Experiment-1/Post_Training_Evals/post_training_comparison.py`

### Experiment 2: SOTA Benchmarking on Data Discovery
Evaluates discovery accuracy on the Pharma Protocol benchmark against state-of-the-art baselines: **CMDL**, **TaBERT**, and **TabSTAR**. Includes candidate pool scalability studies ($K \in [1, 32]$, candidate pools from 50 to full corpus).
- Guide: [`Experiment-2/README.md`](Experiment-2/README.md)
- Primary script: `Experiment-2/SOTA_Evaluation_New/run_comparison_pharma.py`
- Scalability driver: `Experiment-2/SOTA_Evaluation_New/run_scalability_pharma.py`

### Experiment 3: End-to-End Data Integration Vs. Frontier LLM Baselines
Evaluates full relationship-type table materialization across 382 hospital admissions from MIMIC-IV. Compares LOKI against direct prompting on frontier LLMs (Qwen-3.7-Max, Qwen-3.6-Local, GPT-OSS 20B) across table materialization metrics, cluster purity, and token economics.
- Guide: [`Experiment-3/README.md`](Experiment-3/README.md)
- Primary report: [`Experiment-3/#Results/relationship_table_report.md`](Experiment-3/#Results/relationship_table_report.md)
- Compute cost methodology: [`Experiment-3/#Results/Compute_Cost/README.md`](Experiment-3/#Results/Compute_Cost/README.md)

<!-- ---

## Citation

If you use LOKI in your research, please cite:

```bibtex
@article{loki2027vldb,
  title={Discovery-Driven Integration of Disjoint Tables via Text},
  author={...},
  journal={Proceedings of the VLDB Endowment (PVLDB)},
  year={2027}
}
```

--- -->

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE.txt](LICENSE.txt) file for details.
