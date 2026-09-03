# Post-Training 5-Model Comparison - Instructions to Run

## Summary
`post_training_comparison.py` now generates a 5-model comparison:

- `Baseline` (taken from LOKI Stage 0 / frozen encoder)
- `FT-Encoder`
- `Uni (R⟶S)`
- `Uni (S⟶R)`
- `LOKI`

You only need to run post-training evaluation for the 4 trained models. The `Baseline` metrics are loaded from the LOKI evaluation output.

The scripts in `Post_Training_Evals/` now import shared code directly from the sibling `LOKI/` directory. Keep the folder layout as:

```text
<repo-root>/
  Datasets/
  LOKI/
  Post_Training_Evals/
    Input_Models/
      FT-Encoder/
      Uni (R-S)/
      Uni (S-R)/
      LOKI/
    output_plots/
```

### Dataset Setup

Before running post-training evaluation, manually download or copy the
`Datasets/` directory from
[LOKI Datasets](https://github.com/dtim-upc/LOKI/tree/main/Datasets/Datasets).
Place it directly under the repository root and extract any downloaded archive
so the files are available on disk (not left inside a ZIP file).

For the MIMIC post-training workflow, the required layout is:

```text
<repo-root>/
  Datasets/
    mimic/
      Annotated_Test.json
      test_row_level.json
```

The directory name must be exactly `mimic`. Only `Annotated_Test.json` and
`test_row_level.json` are required for post-training row-sentence evaluation.
`train_row_level.json` and `val_row_level.json` are optional: when present,
the script includes their descriptive statistics in the output; when absent,
evaluation continues after a warning. They are required only to train or
validate models, not to evaluate saved checkpoints.

The trained model files are not included in the GitHub repository. Before
running any evaluation or comparison, install `huggingface_hub` and download
the published 10.6 GB model set into `Input_Models/`. Each downloaded folder
includes its `args.json`, model checkpoint files, and `training_data/` directory
when available. `output_plots/` is reserved for generated training-curve plots
and `combined_comparison_data.json`.

From inside `Post_Training_Evals/`:

```powershell
pip install huggingface_hub
python model_download.py
```

From the repository root, specify the `Post_Training_Evals` directory and the
destination explicitly:

```powershell
pip install huggingface_hub
python Post_Training_Evals/model_download.py --destination Post_Training_Evals/Input_Models
```

To download only one model, provide its exact folder name, for example:

```powershell
python model_download.py --models "FT-Encoder"
```

## Generated Outputs
All outputs are saved to `Post_Training_Comparison_Plots/` in both PNG and PDF formats:

- `post_training_4model_bars`
- `post_training_4model_ranking`
- `post_training_4model_roc_pr`
- `post_training_4model_radar`
- `post_training_4model_dashboard`

Note: the filenames still use the legacy `4model` prefix for backward compatibility, even though the comparison now includes 5 models.

---

## Prerequisites
- Python 3.8+ (or your project’s required version)
- Recommended packages: `matplotlib`, `seaborn`, `numpy`, `scikit-learn`
- The full training/evaluation environment used by `LOKI` (for example `sentence_transformers`, `torch`, etc.)
- Run the commands from inside `Post_Training_Evals/`

## Full Reproduction

### 1) Activate your environment

PowerShell / CMD:
```powershell
conda activate LOKI
```

Or activate your project virtualenv.

### 2) Change into `Post_Training_Evals`

PowerShell:
```powershell
cd Post_Training_Evals
```

CMD:
```cmd
cd Post_Training_Evals
```

### 3) Run post-training evaluation for each trained model

The evaluation script writes deterministic result folders under `Post_Training_Results/`.

Important behavior for unidirectional models:

- `row_to_sentence` runs are saved under `Post_Training_Results/Uni (R-S)`
- `sentence_to_row` runs are saved under `Post_Training_Results/Uni (S-R)`
- legacy `Uni-cross` results can still be read as a fallback for `Uni (R⟶S)`, but new evaluations should use the normalized folders above

PowerShell examples:
```powershell
python post_training_evaluation.py --output_dir "Input_Models/FT-Encoder" --results_dir "Post_Training_Results"
python post_training_evaluation.py --output_dir "Input_Models/Uni (R-S)" --results_dir "Post_Training_Results"
python post_training_evaluation.py --output_dir "Input_Models/Uni (S-R)" --results_dir "Post_Training_Results"
python post_training_evaluation.py --output_dir "Input_Models/LOKI" --results_dir "Post_Training_Results"
```

Add `--download_models` to any evaluation command to download its requested
published model folder automatically when it is absent locally.

CMD examples:
```cmd
python post_training_evaluation.py --output_dir "Input_Models/FT-Encoder" --results_dir "Post_Training_Results"
python post_training_evaluation.py --output_dir "Input_Models/Uni (R-S)" --results_dir "Post_Training_Results"
python post_training_evaluation.py --output_dir "Input_Models/Uni (S-R)" --results_dir "Post_Training_Results"
python post_training_evaluation.py --output_dir "Input_Models/LOKI" --results_dir "Post_Training_Results"
```

After running, you should have:

- `Post_Training_Results/FT-Encoder/results_post_training_eval.json`
- `Post_Training_Results/Uni (R-S)/results_post_training_eval.json`
- `Post_Training_Results/Uni (S-R)/results_post_training_eval.json`
- `Post_Training_Results/LOKI/results_post_training_eval.json`

### 4) Generate comparison plots

Once the evaluation JSON files are present, run the comparison script. It also reads `output_plots/combined_comparison_data.json` when present to override some AP / F1 values.

PowerShell:
```powershell
python post_training_comparison.py --loki_results "Post_Training_Results/LOKI" --ftencoder_dir "Post_Training_Results/FT-Encoder" --uni_rs_dir "Post_Training_Results/Uni (R-S)" --uni_sr_dir "Post_Training_Results/Uni (S-R)"
```

CMD:
```cmd
python post_training_comparison.py --loki_results "Post_Training_Results/LOKI" --ftencoder_dir "Post_Training_Results/FT-Encoder" --uni_rs_dir "Post_Training_Results/Uni (R-S)" --uni_sr_dir "Post_Training_Results/Uni (S-R)"
```

If you omit `--uni_rs_dir` or `--uni_sr_dir`, the comparison script will try to auto-resolve common folder names.

### 5) Generate Training Curves & Emergent Ability Plots

Once the evaluations are complete and `Input_Models/` contains the training data, you can generate the rich suite of training efficiency and emergent ability visualizations (like the 3D emergence cliff and global-to-local transfer plots).

PowerShell / CMD:
```powershell
python training_curves.py --compare
```

This will automatically discover training logs in `Input_Models/`, combine them into `output_plots/combined_comparison_data.json`, and output over a dozen multi-model comparison figures directly into `output_plots/`.

Use `python training_curves.py --compare --download_models` to download any
missing published model folders before generating the plots.

### 6) Optional smoke test on 3-5 examples

If you want to quickly verify that the pipeline runs before launching the full evaluation, you can limit row-sentence evaluation to a very small deterministic subset of annotated examples.

PowerShell:
```powershell
python post_training_evaluation.py --output_dir "Input_Models/FT-Encoder" --results_dir "Post_Training_Results" --quick --skip_pair_scores --no_plots --row_sent_max_examples 5
```

CMD:
```cmd
python post_training_evaluation.py --output_dir "Input_Models/FT-Encoder" --results_dir "Post_Training_Results" --quick --skip_pair_scores --no_plots --row_sent_max_examples 5
```

Notes:

- `--row_sent_max_examples 5` limits evaluation to 5 row-sentence test examples
- the subset is deterministic and prefers examples that match the annotation file
- `--quick` skips the Stage 2 pretraining-style evaluation
- `--skip_pair_scores` and `--no_plots` make the smoke test much faster and smaller

## Optional Flags
- `--combined_data`: path to `combined_comparison_data.json` (default: `output_plots/combined_comparison_data.json`)
- `--output_dir`: output directory for plots (default: `Post_Training_Comparison_Plots`)
- `--stage_priority`: comma-separated Stage-3 priority order, for example `stage_3_best_test_overall_acc,stage_3_best_test_avg_precision,stage_3_best`
- `--unicross_dir`: deprecated alias for `--uni_rs_dir`
- `--row_sent_max_examples`: maximum number of row-sentence test examples to evaluate in `post_training_evaluation.py`
- `--quick`: skip Stage 2 and only run Stage 0 plus trained checkpoints
- `--skip_pair_scores`: skip writing detailed row-sentence pair scores
- `--no_plots`: skip plot generation during evaluation
- `--download_models`: download missing published model folders from Hugging Face

## Where Outputs Go
- All generated figures are written to the directory specified by `--output_dir`
- `comparison_metrics.json` is saved in the same folder
- `comparison_raw_counts.json` is also saved in the same folder for ranking / prediction-count inspection
- `dataset_statistics.json` is saved from the LOKI evaluation when available

## Troubleshooting
- If plotting libraries fail to import, install the required packages in the active environment
- If `post_training_evaluation.py` fails on imports, make sure you are using the same Python environment as `LOKI`
- If ROC/PR curves are blank, verify that `pair_scores_data` exists in each model’s `results_post_training_eval.json`
- If a unidirectional model does not show up, confirm that its evaluation wrote to `Post_Training_Results/Uni (R-S)` or `Post_Training_Results/Uni (S-R)`
- If your unidirectional run folders differ from the examples above, replace the `--output_dir` values with the actual folders that contain `args.json`
- If you only want to confirm that the script runs end-to-end, start with `--row_sent_max_examples 3` or `--row_sent_max_examples 5`

## Optional Minimal Requirements
To make reproduction easier, a minimal plotting-only environment would include:

```text
matplotlib
seaborn
numpy
scikit-learn
```

But for `post_training_evaluation.py`, you should use the full LOKI environment rather than a plotting-only environment.
