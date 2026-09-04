# Experiment-2: TaBERT Fine-tuning & Evaluation

## Prerequisites

- **Conda environment:** `LOKI`
- **GPU:** NVIDIA GPU with ≥6 GB VRAM
- **Python packages:** `torch`, `transformers`, `peft`, `accelerate`, `numpy`, `tqdm`

Activate the environment:
```powershell
conda activate THOR
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

## Directory Structure

All paths are relative to `TaBERT/`:

```
TaBERT/
├── pretrained/
│   └── tabert_large_k3/
│       ├── model.bin              # Pretrained TaBERT weights (required)
│       └── tb_config.json         # Model config
├── pharma_flipped_structured/     # Dataset directory (required)
│   ├── train_row_level.json       # Training split
│   ├── val_row_level.json         # Validation split
│   ├── test_row_level.json        # Test split
│   └── Annotated_Test.json        # Row-sentence annotations (for --run_row_sentence)
├── finetune.py                    # Training script
├── evaluate_finetuned.py          # Evaluation script
├── data_loader.py                 # Data pipeline
└── model_wrapper.py               # LoRA model wrapper
```

**Dataset:** Place the `pharma_flipped_structured/` folder inside the `TaBERT/` directory.
Each JSON file contains a list of examples with keys: `anchor_id`, `anchor_sentences` (list of NL strings), `primary_positive`, `additional_positives`, `negatives`, `threshold`.
Table items (positives/negatives) carry structured data: `headers` (column names) and `rows` (each with `row_idx`, `content` array, and `formatted` string).

## Fine-tuning

### Default run (recommended)
```powershell
python finetune.py
```

This uses all default settings: 3 epochs, batch size 2 × 8 gradient accumulation = 16 effective, LoRA r=16, margin=0.3.

### Custom run
```powershell
python finetune.py --epochs 5 --batch_size 4 --learning_rate 1e-4 --lora_r 32
```

### All training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `pharma_flipped_structured` | Dataset directory |
| `--model_path` | `pretrained/tabert_large_k3/model.bin` | Pretrained model weights |
| `--output_dir` | `finetuned_tabert_pharma` | Output directory for checkpoints |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `2` | Per-step batch size |
| `--grad_accum_steps` | `8` | Gradient accumulation steps (effective batch = batch_size × this) |
| `--learning_rate` | `2e-4` | Peak learning rate |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--max_grad_norm` | `1.0` | Gradient clipping |
| `--warmup_ratio` | `0.1` | Fraction of steps for LR warmup |
| `--margin` | `0.3` | Triplet loss margin |
| `--scale` | `10.0` | Triplet loss scale |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--sample_row_num` | `3` | Table rows per forward pass (0 = all rows) |
| `--max_context_len` | `128` | Max sentence tokens during training |
| `--triplet_strategy` | `limited` | Triplet generation: `primary_only`, `limited`, `random`, `full` |
| `--max_triplets_per_example` | `10` | Max triplets per example |
| `--use_amp` | `True` | Mixed precision training |
| `--seed` | `42` | Random seed |

### Training Output

After training, the output directory (`finetuned_tabert_pharma/` by default) contains:

```
finetuned_tabert_pharma/
├── best/                    # Best model by validation accuracy ← USE THIS
│   ├── model_merged.bin     # Full merged weights (LoRA folded in)
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── extra_state.pt       # Temperature parameter
├── final/                   # Model after last epoch
│   ├── model_merged.bin
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── extra_state.pt
├── epoch_1/                 # Per-epoch LoRA checkpoints (adapters only)
├── epoch_2/
├── epoch_3/
└── training_log.json        # Training metrics per epoch + config
```

**Use the `best/` checkpoint** for evaluation — it has the highest validation accuracy.

## Evaluation

### Table-level evaluation (default)
```powershell
python evaluate_finetuned.py --model_dir finetuned_tabert_pharma/best
```

### With row-sentence grounding
```powershell
python evaluate_finetuned.py --model_dir finetuned_tabert_pharma/best --run_row_sentence
```

### With frozen baseline comparison
```powershell
python evaluate_finetuned.py --model_dir finetuned_tabert_pharma/best --run_frozen_baseline
```

### Full evaluation (all modes)
```powershell
python evaluate_finetuned.py --model_dir finetuned_tabert_pharma/best --run_row_sentence --run_frozen_baseline
```

### Evaluate on validation split instead of test
```powershell
python evaluate_finetuned.py --model_dir finetuned_tabert_pharma/best --split val
```

### All evaluation parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_dir` | `finetuned_tabert_pharma/best` | Fine-tuned model checkpoint |
| `--base_model_path` | `pretrained/tabert_large_k3/model.bin` | Base pretrained model |
| `--data_dir` | `pharma_flipped_structured` | Dataset directory |
| `--split` | `test` | Split to evaluate (`test` or `val`) |
| `--sample_row_num` | `3` | Table rows for table-level eval (0 = all rows) |
| `--max_context_len` | `512` | Max sentence tokens during evaluation |
| `--output_file` | `<model_dir>/eval_results.json` | Where to save results JSON |
| `--run_frozen_baseline` | off | Also evaluate pretrained model without fine-tuning |
| `--run_row_sentence` | off | Also run row-sentence P/R/F1 evaluation |

### Evaluation Output

Results are saved to `eval_results.json` inside the model directory. Metrics include:

- **Table-level:** `table_level_accuracy`, `mean_positive_score`, `mean_negative_score`, `score_separation`
- **Row-sentence** (if `--run_row_sentence`): `row_sent_precision`, `row_sent_recall`, `row_sent_f1`
- **Frozen baseline** (if `--run_frozen_baseline`): same metrics prefixed with `frozen_`

## Table Serialization

TaBERT constructs multi-column `Table` objects directly from the structured `headers` and `rows[].content` arrays in the `pharma_flipped_structured` dataset. Each column header becomes a proper TaBERT `Column`, and each row's content array provides that row's cell values — no regex parsing of formatted strings is needed. This is the architecturally correct mode for TaBERT's vertical self-attention, as each column gets its own representation.

## Using the Fine-tuned Model in SOTA Evaluation

To use your fine-tuned TaBERT in the `SOTA_Evaluation_New/` comparison framework, copy the merged weights into the SOTA models folder, renaming the file to `model.bin`:

```powershell
copy "finetuned_tabert_pharma\best\model_merged.bin" `
     "..\SOTA_Evaluation_New\models\TaBERT\tabert_large_k3\model.bin"
```

This replaces the original pretrained `model.bin` with your fine-tuned weights. Leave all other files in the folder (e.g., `tb_config.json`) unchanged. The SOTA evaluation scripts will then use the fine-tuned model automatically.

## Quick Start (copy-paste)

```powershell
conda activate THOR
$env:KMP_DUPLICATE_LIB_OK="TRUE"
cd TaBERT

# Train
python finetune.py

# Evaluate best model
python evaluate_finetuned.py --model_dir finetuned_tabert_pharma/best --run_row_sentence --run_frozen_baseline
```


## Reference

If you plan to use `TaBERT` in your project, please consider citing [our paper](https://arxiv.org/abs/2005.08314):
```
@inproceedings{yin20acl,
    title = {Ta{BERT}: Pretraining for Joint Understanding of Textual and Tabular Data},
    author = {Pengcheng Yin and Graham Neubig and Wen-tau Yih and Sebastian Riedel},
    booktitle = {Annual Conference of the Association for Computational Linguistics (ACL)},
    month = {July},
    year = {2020}
}
```