"""
metrics.py — Shared ranking metrics for Table-Text discovery evaluation.

Provides two families of Average Precision:
  1. Rank-based AP / MAP  —  from discrete ranked lists (standard IR)
  2. Score-based AP (PR-curve)  —  from continuous similarity scores (sklearn)

All rank-based functions operate on **table IDs** (not embeddings).
Ground truth and predictions are dictionaries: query_id -> [table_id, ...]
"""

import numpy as np
from typing import Dict, List, Any


# ===========================================================================
# Per-query rank-based metrics
# ===========================================================================

def precision_at_k(gt_tables: List[str], ranked_tables: List[str], k: int) -> float:
    """Precision@K: fraction of top-K predictions that are relevant."""
    if k <= 0:
        return 0.0
    top_k = ranked_tables[:k]
    if not top_k:
        return 0.0
    gt_set = set(gt_tables)
    hits = sum(1 for t in top_k if t in gt_set)
    return hits / len(top_k)


def recall_at_k(gt_tables: List[str], ranked_tables: List[str], k: int) -> float:
    """Recall@K: fraction of relevant tables found in top-K."""
    if k <= 0 or not gt_tables:
        return 0.0
    top_k = ranked_tables[:k]
    gt_set = set(gt_tables)
    hits = sum(1 for t in top_k if t in gt_set)
    return hits / len(gt_set)


def f1_at_k(gt_tables: List[str], ranked_tables: List[str], k: int) -> float:
    """F1@K: harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(gt_tables, ranked_tables, k)
    r = recall_at_k(gt_tables, ranked_tables, k)
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def average_precision(gt_tables: List[str], ranked_tables: List[str]) -> float:
    """
    Average Precision (AP) for a single query — rank-based.

    AP = (1/|relevant|) * sum_{k=1}^{N} P(k) * rel(k)
    """
    if not gt_tables:
        return 0.0
    gt_set = set(gt_tables)
    num_relevant = len(gt_set)
    hits = 0
    sum_precision = 0.0
    for i, table in enumerate(ranked_tables):
        if table in gt_set:
            hits += 1
            sum_precision += hits / (i + 1)
    if num_relevant == 0:
        return 0.0
    return sum_precision / num_relevant


def ndcg_at_k(gt_tables: List[str], ranked_tables: List[str], k: int) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain at K."""
    if k <= 0 or not gt_tables:
        return 0.0
    gt_set = set(gt_tables)
    top_k = ranked_tables[:k]
    dcg = 0.0
    for i, table in enumerate(top_k):
        if table in gt_set:
            dcg += 1.0 / np.log2(i + 2)
    num_relevant = min(k, len(gt_set))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(gt_tables: List[str], ranked_tables: List[str], k: int) -> float:
    """MRR@K: Reciprocal of the rank of the first relevant item in top-K."""
    if k <= 0 or not gt_tables:
        return 0.0
    gt_set = set(gt_tables)
    for rank, table in enumerate(ranked_tables[:k], 1):
        if table in gt_set:
            return 1.0 / rank
    return 0.0


def all_at_k(gt_tables: List[str], ranked_tables: List[str], k: int) -> float:
    """All@K: 1.0 if ALL ground truth tables appear in the top-K, else 0.0.

    Measures complete retrieval — whether the model found every relevant
    table within the top-K results. Naturally 0 when K < |GT|.
    """
    if k <= 0 or not gt_tables:
        return 0.0
    gt_set = set(gt_tables)
    top_k_set = set(ranked_tables[:k])
    return 1.0 if gt_set.issubset(top_k_set) else 0.0


def mean_rank(gt_tables: List[str], ranked_tables: List[str]) -> float:
    """Mean rank of all relevant tables (1-indexed, lower is better)."""
    if not gt_tables:
        return float("inf")
    gt_set = set(gt_tables)
    ranks = []
    for rank, table in enumerate(ranked_tables, 1):
        if table in gt_set:
            ranks.append(rank)
    return np.mean(ranks) if ranks else float("inf")


# ===========================================================================
# Score-based AP (from PR curve — matches post-training evals)
# ===========================================================================

def score_based_average_precision(
    gt_map: Dict[str, List[str]],
    scores_map: Dict[str, Dict[str, float]],
) -> float:
    """
    Compute Average Precision from continuous scores using sklearn.

    This matches the AP computation in post-training evals
    (sklearn.metrics.average_precision_score), which computes the area
    under the Precision-Recall curve.

    Args:
        gt_map:     query_id -> [relevant_table_id, ...]
        scores_map: query_id -> {table_id: similarity_score, ...}

    Returns:
        Score-based AP (area under PR curve), averaged over all queries.
    """
    from sklearn.metrics import average_precision_score

    query_ids = sorted(set(gt_map.keys()) & set(scores_map.keys()))
    if not query_ids:
        return 0.0

    all_ap = []
    for qid in query_ids:
        gt_set = set(gt_map[qid])
        table_scores = scores_map[qid]

        if not table_scores or not gt_set:
            all_ap.append(0.0)
            continue

        table_ids = sorted(table_scores.keys())
        y_true = np.array([1 if tid in gt_set else 0 for tid in table_ids])
        y_scores = np.array([table_scores[tid] for tid in table_ids])

        if len(np.unique(y_true)) > 1:
            all_ap.append(float(average_precision_score(y_true, y_scores)))
        else:
            # All same class -> AP is 1.0 if all relevant, 0.0 if none
            all_ap.append(float(y_true[0]))

    return float(np.mean(all_ap))


# ===========================================================================
# Aggregate evaluation
# ===========================================================================

def evaluate_retrieval(
    gt_map: Dict[str, List[str]],
    predictions_map: Dict[str, List[str]],
    k_values: List[int],
    scores_map: Dict[str, Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate retrieval over all queries.

    Args:
        gt_map: query_id -> list of relevant table_ids
        predictions_map: query_id -> ranked list of table_ids (best first)
        k_values: list of K values for @K metrics
        scores_map: (optional) query_id -> {table_id: score} for PR-curve AP

    Returns:
        Dictionary with per-K metrics, MAP (rank-based), Score_AP (PR-curve),
        and Mean Rank.
    """
    query_ids = sorted(set(gt_map.keys()) & set(predictions_map.keys()))

    if not query_ids:
        empty_per_k = {}
        for k in k_values:
            empty_per_k[k] = {"P@K": 0, "R@K": 0, "F1@K": 0, "NDCG@K": 0, "MRR@K": 0, "All@K": 0}
        return {
            "num_queries": 0,
            "per_k": empty_per_k,
            "MAP": 0.0,
            "Score_AP": 0.0,
            "Mean_Rank": float("inf"),
        }

    # Accumulators
    base_metrics = ["P@K", "R@K", "F1@K", "NDCG@K", "MRR@K"]
    per_k_metrics = {k: {m: [] for m in base_metrics} for k in k_values}
    all_k_metrics = {k: [] for k in k_values}
    all_ap = []
    all_mean_rank = []

    for qid in query_ids:
        gt = gt_map[qid]
        pred = predictions_map[qid]

        all_ap.append(average_precision(gt, pred))
        all_mean_rank.append(mean_rank(gt, pred))

        for k in k_values:
            per_k_metrics[k]["P@K"].append(precision_at_k(gt, pred, k))
            per_k_metrics[k]["R@K"].append(recall_at_k(gt, pred, k))
            per_k_metrics[k]["F1@K"].append(f1_at_k(gt, pred, k))
            per_k_metrics[k]["NDCG@K"].append(ndcg_at_k(gt, pred, k))
            per_k_metrics[k]["MRR@K"].append(mrr_at_k(gt, pred, k))
            all_k_metrics[k].append(all_at_k(gt, pred, k))

    # Average across queries
    result = {
        "num_queries": len(query_ids),
        "per_k": {},
        "MAP": float(np.mean(all_ap)),
        "Mean_Rank": float(np.mean(all_mean_rank)),
    }
    for k in k_values:
        result["per_k"][k] = {
            metric: float(np.mean(values))
            for metric, values in per_k_metrics[k].items()
        }
        result["per_k"][k]["All@K"] = float(np.mean(all_k_metrics[k]))

    # Score-based AP (PR-curve) if scores are provided
    if scores_map is not None:
        result["Score_AP"] = score_based_average_precision(gt_map, scores_map)
    else:
        result["Score_AP"] = None

    return result


# ===========================================================================
# Micro-averaged metrics (matches run_pharma_cmdl.py eval_matches)
# ===========================================================================

def eval_matches_micro(
    gt_map: Dict[str, List[str]],
    predictions: Dict[str, List[str]],
) -> tuple:
    """
    Micro-averaged P/R/F1 — matches run_pharma_cmdl.py eval_matches.

    Pools TP/FP/FN globally across all queries, then computes single
    precision, recall, and F1 values. Each matching pair contributes
    equally regardless of which query it belongs to.

    Args:
        gt_map: query_id -> list of relevant table_ids
        predictions: query_id -> list of predicted table_ids (already truncated to K)

    Returns:
        (precision, recall, f1) tuple
    """
    tp = 0
    fp = 0
    fn = 0
    for idx in predictions:
        gt_matches = set(gt_map.get(idx, []))
        if len(gt_matches) == 0:
            continue
        pred_matches = set(predictions.get(idx, []))
        true_matches = len(gt_matches.intersection(pred_matches))
        false_matches = len(pred_matches) - true_matches
        non_matches = len(gt_matches) - true_matches
        tp += true_matches
        fp += false_matches
        fn += non_matches
    fp = 1 if (tp + fp) == 0 else fp
    prec = 1.0 * tp / (tp + fp)
    rec = 1.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0 if tp == 0 else 2.0 * prec * rec / (prec + rec)
    return prec, rec, f1


def score_based_average_precision_micro(
    gt_map: Dict[str, List[str]],
    scores_map: Dict[str, Dict[str, float]],
) -> tuple:
    """
    Compute Score-based Average Precision & Rank-based MAP by pooling all 
    (query, table) pairs globally.

    For micro AP/MAP, we flatten all query-table combinations into a single array 
    and evaluate ranking metrics on the entire set at once.

    Args:
        gt_map:     query_id -> [relevant_table_id, ...]
        scores_map: query_id -> {table_id: similarity_score, ...}

    Returns:
        (score_ap, rank_map, mean_rank) for the global pool.
    """
    from sklearn.metrics import average_precision_score

    query_ids = sorted(set(gt_map.keys()) & set(scores_map.keys()))
    if not query_ids:
        return 0.0, 0.0, float('inf')

    # Flatten out arrays
    global_y_true = []
    global_y_scores = []
    
    # We will compute pseudo rank-based metrics manually
    # by sorting the entire pool
    pooled_pairs = []

    for qid in query_ids:
        gt_set = set(gt_map[qid])
        table_scores = scores_map[qid]

        if not table_scores:
            continue
            
        for tid, score in table_scores.items():
            label = 1 if tid in gt_set else 0
            global_y_true.append(label)
            global_y_scores.append(score)
            pooled_pairs.append((score, label))

    if not global_y_true:
        return 0.0, 0.0, float('inf')

    global_y_true = np.array(global_y_true)
    global_y_scores = np.array(global_y_scores)

    # Scikit-learn AP (PR Curve Area)
    if len(np.unique(global_y_true)) > 1:
        score_ap = float(average_precision_score(global_y_true, global_y_scores))
    else:
        score_ap = float(global_y_true[0])

    # Rank-based metrics (MAP and Mean Rank) over the global pool
    pooled_pairs.sort(key=lambda x: x[0], reverse=True)
    
    hits = 0
    sum_precision = 0.0
    ranks = []
    
    num_relevant = sum(global_y_true)
    
    for i, (score, label) in enumerate(pooled_pairs):
        if label == 1:
            hits += 1
            sum_precision += hits / (i + 1)
            ranks.append(i + 1)
            
    rank_map = sum_precision / num_relevant if num_relevant > 0 else 0.0
    m_rank = float(np.mean(ranks)) if ranks else float('inf')

    return score_ap, rank_map, m_rank


def evaluate_retrieval_micro(
    gt_map: Dict[str, List[str]],
    predictions_map: Dict[str, List[str]],
    k_values: List[int],
    scores_map: Dict[str, Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate retrieval using micro-averaged P/R/F1@K.

    For each K, truncates each query's predictions to the top K, then
    computes global micro-averaged precision, recall, and F1.

    Args:
        gt_map: query_id -> list of relevant table_ids
        predictions_map: query_id -> ranked list of table_ids (best first)
        k_values: list of K values for @K metrics

    Returns:
        Dictionary with per-K micro-averaged metrics and num_queries.
    """
    query_ids = sorted(set(gt_map.keys()) & set(predictions_map.keys()))

    if not query_ids:
        empty_per_k = {}
        for k in k_values:
            empty_per_k[k] = {"P@K": 0, "R@K": 0, "F1@K": 0}
        return {"num_queries": 0, "per_k": empty_per_k, "metric_type": "micro"}

    # Only keep queries present in both maps
    filtered_predictions = {qid: predictions_map[qid] for qid in query_ids}
    filtered_gt = {qid: gt_map[qid] for qid in query_ids}

    result = {
        "num_queries": len(query_ids),
        "per_k": {},
        "metric_type": "micro",
    }
    for k in k_values:
        truncated = {qid: tables[:k] for qid, tables in filtered_predictions.items()}
        p, r, f1 = eval_matches_micro(filtered_gt, truncated)
        result["per_k"][k] = {"P@K": p, "R@K": r, "F1@K": f1}

    # Include pool-based AP/Rank metrics if scores were provided
    if scores_map is not None:
        score_ap, rank_map, m_rank = score_based_average_precision_micro(filtered_gt, {q: scores_map[q] for q in query_ids})
        result["MAP"] = rank_map
        result["Score_AP"] = score_ap
        result["Mean_Rank"] = m_rank
    else:
        result["MAP"] = 0.0
        result["Score_AP"] = None
        result["Mean_Rank"] = float("inf")

    return result


def print_results_table_micro(results: Dict[str, Any], model_name: str = "Model"):
    """Pretty-print micro-averaged evaluation results."""
    print(f"\n{'=' * 55}")
    print(f"  {model_name} — Micro-Averaged Results")
    print(f"  Evaluated {results['num_queries']} queries")
    print(f"{'=' * 55}")
    print(f"  {'K':<5}  {'P@K':<10}  {'R@K':<10}  {'F1@K':<10}")
    print(f"  {'-' * 40}")
    for k, metrics in sorted(results["per_k"].items(), key=lambda x: int(x[0])):
        print(f"  {k:<5}  {metrics['P@K']:<10.4f}  {metrics['R@K']:<10.4f}  "
              f"{metrics['F1@K']:<10.4f}")
    print(f"  {'-' * 40}")
    
    if "MAP" in results:
        print(f"  MAP (rank-based):   {results['MAP']:.4f}")
    if results.get("Score_AP") is not None:
        print(f"  AP  (PR-curve):     {results['Score_AP']:.4f}")
    if "Mean_Rank" in results:
        print(f"  Mean Rank:          {results['Mean_Rank']:.2f}")
        
    print(f"{'=' * 55}\n")


def print_results_table(results: Dict[str, Any], model_name: str = "Model"):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 82}")
    print(f"  {model_name} — Table-Text Discovery Results")
    print(f"  Evaluated {results['num_queries']} queries")
    print(f"{'=' * 82}")
    print(f"  {'K':<5}  {'P@K':<10}  {'R@K':<10}  {'F1@K':<10}  {'NDCG@K':<10}  {'MRR@K':<10}  {'All@K (Hit Rate)':<18}")
    print(f"  {'-' * 77}")
    for k, metrics in sorted(results["per_k"].items(), key=lambda x: x[0]):
        all_str = f"{metrics['All@K']:<18.4f}" if "All@K" in metrics else "—".ljust(18)
        print(f"  {k:<5}  {metrics['P@K']:<10.4f}  {metrics['R@K']:<10.4f}  "
              f"{metrics['F1@K']:<10.4f}  {metrics['NDCG@K']:<10.4f}  {metrics['MRR@K']:<10.4f}  "
              f"{all_str}")
    print(f"  {'-' * 77}")
    print(f"  MAP (rank-based):   {results['MAP']:.4f}")
    if results.get("Score_AP") is not None:
        print(f"  AP  (PR-curve):     {results['Score_AP']:.4f}")
    print(f"  Mean Rank:          {results['Mean_Rank']:.2f}")
    print(f"{'=' * 82}\n")
