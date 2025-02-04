import os
import json
import pytrec_eval
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Define the directory and input files
INPUT_DIR = "input"
OUTPUT_DIR = "output"
INPUT_FILES = ["full_test_output.json", "full_validation_output.json"]

# K values to evaluate
K_VALUES = [1, 3, 5, 10]

def get_metrics_for_k(k):
    """Return a set of metrics for a given cutoff value k.
    Note: pytrec_eval returns metric names with underscores (e.g., "P_10")
    """
    return {
        f'P.{k}',
        f'recall.{k}',
        f'map_cut.{k}',
        f'ndcg_cut.{k}',
    }

def load_data(file_path):
    """
    Loads the JSON data from the given file path.
    Expects a list of anchors where each anchor has:
      - "anchor_id"
      - "predictions": list of dicts with candidate_id, cosine_similarity, relevance
      - "ground_truth": list of dicts with candidate_id, distance, relevance
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def build_qrel_run(data):
    """
    Builds the qrel and run dictionaries in the format that pytrec_eval expects.
    Keys (anchor ids and candidate ids) are cast to strings.
    
    qrel: { anchor_id: { candidate_id: ground_truth_relevance, ... }, ... }
    run:  { anchor_id: { candidate_id: cosine_similarity, ... }, ... }
    """
    qrel = {}
    run = {}
    for anchor in data:
        anchor_id = str(anchor["anchor_id"])
        # Build ground truth relevance judgments from ground_truth list.
        qrel[anchor_id] = {
            str(item["candidate_id"]): item["relevance"]
            for item in anchor["ground_truth"]
        }
        # Build run dictionary from predictions using cosine_similarity as the score.
        run[anchor_id] = {
            str(item["candidate_id"]): item["cosine_similarity"]
            for item in anchor["predictions"]
        }
    return qrel, run

def evaluate_for_k(qrel, run, k):
    """
    Evaluates the qrel and run dictionaries using pytrec_eval for a given k.
    Returns:
      - detailed_results: A dictionary with anchor ids as keys and metric values as values.
      - aggregated: A dictionary with the average score per metric (over all anchors).
    """
    metrics = get_metrics_for_k(k)
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
    detailed_results = evaluator.evaluate(run)
    
    # Aggregate (average) the scores for each metric across all queries (anchors)
    agg = defaultdict(list)
    for query_results in detailed_results.values():
        for metric, score in query_results.items():
            agg[metric].append(score)
    aggregated = {metric: sum(scores) / len(scores) for metric, scores in agg.items()}
    
    return detailed_results, aggregated

def evaluate_file(file_path, k_values):
    """
    Loads the data from the file, builds the qrel and run dictionaries,
    and evaluates for each K in k_values.
    
    Returns a dictionary with K values as keys and a tuple of (detailed_results, aggregated)
    """
    data = load_data(file_path)
    qrel, run = build_qrel_run(data)
    
    evaluation_results = {}
    for k in k_values:
        detailed, aggregated = evaluate_for_k(qrel, run, k)
        evaluation_results[k] = {
            "detailed": detailed,
            "aggregated": aggregated
        }
    return evaluation_results

def print_and_save_results(file_name, evaluation_results):
    """
    Prints the aggregated metrics to the console and saves the results as an Excel file.
    The Excel file will have separate sheets for each K value (detailed results) 
    as well as one sheet for the aggregated metrics with columns:
    K, P@K, R@K, MAP@K, NDCG@K.
    """
    print(f"\nEvaluation Results for {file_name}:")

    # Set up directories
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)    
    
    output_excel = output_path / f"{os.path.splitext(file_name)[0]}_evaluation.xlsx"
    
    # Use the context manager for ExcelWriter
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # List to collect aggregated records with desired column names.
        agg_records = []
        for k, results in evaluation_results.items():
            agg = results["aggregated"]
            # Update keys to match pytrec_eval output: underscores instead of dots.
            record = {
                "K": k,
                "P@K": agg.get(f'P_{k}', None),
                "R@K": agg.get(f'recall_{k}', None),
                "MAP@K": agg.get(f'map_cut_{k}', None),
                "NDCG@K": agg.get(f'ndcg_cut_{k}', None)
            }
            agg_records.append(record)
            
            # Prepare detailed results per query for this K.
            detailed = results["detailed"]
            detailed_records = []
            for query, metrics_dict in detailed.items():
                record_detail = {"anchor_id": query, "K": k}
                record_detail.update(metrics_dict)
                detailed_records.append(record_detail)
            df_detailed = pd.DataFrame(detailed_records)
            df_detailed.sort_values(by="anchor_id", inplace=True)
            sheet_name = f"K_{k}_detailed"
            df_detailed.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Create the aggregated DataFrame with the required columns and no nulls.
        df_agg = pd.DataFrame(agg_records)
        df_agg.sort_values(by="K", inplace=True)
        df_agg.set_index("K", inplace=True)
        df_agg = df_agg[["P@K", "R@K", "MAP@K", "NDCG@K"]]
        df_agg.to_excel(writer, sheet_name="Aggregated")
    
    # Print aggregated results to console
    print(df_agg)
    print(f"Saved Excel evaluation results to {output_excel}")

def main():
    for file_name in INPUT_FILES:
        file_path = os.path.join(INPUT_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        print(f"Evaluating file: {file_path}")
        evaluation_results = evaluate_file(file_path, K_VALUES)
        print_and_save_results(file_name, evaluation_results)

if __name__ == "__main__":
    main()
