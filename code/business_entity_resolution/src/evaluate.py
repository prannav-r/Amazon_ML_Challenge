"""
S1-Level Macro-F0.5 Evaluation Module (Phase 5)
Amazon ML Challenge 2026: Business Entity Resolution

Implements the official entity-level evaluation metric:
- Macro-averaged F0.5 across Source 1 entities:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Exact zero-match semantics:
    true empty + predicted empty -> F0.5 = 1.0, Precision = 1.0, Recall = 1.0
    true empty + false predicted match -> F0.5 = 0.0, Precision = 0.0, Recall = 0.0
    true non-empty + predicted empty -> F0.5 = 0.0, Precision = 0.0, Recall = 0.0
- Supports singleton, zero-match, and multi-match diagnostic groupings.
- Supports pair-level diagnostic metrics (ROC-AUC, PR-AUC, LogLoss, Brier score).
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss


def compute_s1_metrics(true_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (precision, recall, f05) for a single Source 1 entity.
    
    Handles zero-match entities according to the official challenge semantics:
      - true empty + predicted empty -> P=1.0, R=1.0, F0.5=1.0
      - true empty + false predicted match -> P=0.0, R=0.0, F0.5=0.0
      - true non-empty + predicted empty -> P=0.0, R=0.0, F0.5=0.0
    """
    num_true = len(true_ids)
    num_pred = len(pred_ids)

    if num_true == 0:
        if num_pred == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    if num_pred == 0:
        return 0.0, 0.0, 0.0

    tp = len(true_ids & pred_ids)
    p = tp / num_pred
    r = tp / num_true

    denom = 0.25 * p + r
    if denom == 0.0 or tp == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * p * r) / denom

    return p, r, f05


def evaluate_s1_macro(
    s1_predictions: Dict[str, Set[str]],
    s1_ground_truth: Dict[str, Set[str]],
    s1_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes macro precision, macro recall, and macro F0.5 across all S1 entities.
    
    Args:
        s1_predictions: mapping of s1_id -> set of predicted candidate entity IDs
        s1_ground_truth: mapping of s1_id -> set of true matched candidate entity IDs
        s1_list: list of S1 entity IDs to evaluate over. If None, evaluates over all S1 in ground truth.
    
    Returns:
        dict with macro_f05, macro_precision, macro_recall, total_s1,
        empty_pred_pct, avg_pred_matches, total_pred_matches.
    """
    if s1_list is None:
        s1_list = sorted(list(s1_ground_truth.keys()))

    total_s1 = len(s1_list)
    if total_s1 == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "total_s1": 0,
            "empty_pred_pct": 0.0,
            "avg_pred_matches": 0.0,
            "total_pred_matches": 0,
        }

    precisions = []
    recalls = []
    f05s = []
    total_matches = 0
    empty_preds = 0

    for s1 in s1_list:
        true_ids = s1_ground_truth.get(s1, set())
        pred_ids = s1_predictions.get(s1, set())

        p, r, f05 = compute_s1_metrics(true_ids, pred_ids)
        precisions.append(p)
        recalls.append(r)
        f05s.append(f05)

        cnt = len(pred_ids)
        total_matches += cnt
        if cnt == 0:
            empty_preds += 1

    return {
        "macro_f05": float(np.mean(f05s)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "total_s1": total_s1,
        "empty_pred_pct": float(empty_preds / total_s1 * 100.0),
        "avg_pred_matches": float(total_matches / total_s1),
        "total_pred_matches": int(total_matches),
    }


def evaluate_predictions_df(
    df_candidate_scores: pd.DataFrame,
    s1_ground_truth: Dict[str, Set[str]],
    threshold: float,
    s1_list: Optional[List[str]] = None,
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
    score_col: str = "score",
) -> Dict[str, Any]:
    """
    Evaluates S1-level macro F0.5 from a DataFrame of scored candidate pairs.
    Applies the decision rule: candidate is matched iff score >= threshold.
    
    Permits multiple candidate matches per S1 entity if multiple exceed threshold.
    Zero-match entities are included in s1_list even if they have zero candidates.
    """
    if s1_list is None:
        s1_list = sorted(list(s1_ground_truth.keys()))

    # Filter by threshold
    mask = df_candidate_scores[score_col] >= threshold
    df_matched = df_candidate_scores[mask]

    # Aggregate predictions by S1
    s1_predictions: Dict[str, Set[str]] = {s1: set() for s1 in s1_list}
    if not df_matched.empty:
        grouped = df_matched.groupby(s1_col)[cand_col].apply(set).to_dict()
        for s1, cands in grouped.items():
            if s1 in s1_predictions:
                s1_predictions[s1] = cands

    return evaluate_s1_macro(s1_predictions, s1_ground_truth, s1_list=s1_list)


def evaluate_by_group(
    df_candidate_scores: pd.DataFrame,
    s1_ground_truth: Dict[str, Set[str]],
    threshold: float,
    s1_list: List[str],
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
    score_col: str = "score",
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates performance partitioned into three diagnostic groups:
      - Group 1: True Singleton (exactly 1 true match)
      - Group 2: True Zero-Match / Empty (0 true matches)
      - Group 3: True Multi-Match (>= 2 true matches)
    """
    group1_s1 = [s1 for s1 in s1_list if len(s1_ground_truth.get(s1, set())) == 1]
    group2_s1 = [s1 for s1 in s1_list if len(s1_ground_truth.get(s1, set())) == 0]
    group3_s1 = [s1 for s1 in s1_list if len(s1_ground_truth.get(s1, set())) > 1]

    # Pre-build predictions
    mask = df_candidate_scores[score_col] >= threshold
    df_matched = df_candidate_scores[mask]
    pred_dict: Dict[str, Set[str]] = {}
    if not df_matched.empty:
        pred_dict = df_matched.groupby(s1_col)[cand_col].apply(set).to_dict()

    s1_predictions: Dict[str, Set[str]] = {s1: pred_dict.get(s1, set()) for s1 in s1_list}

    def _eval_subset(subset: List[str], is_zero_match_group: bool = False) -> Dict[str, Any]:
        if not subset:
            return {"count": 0, "macro_f05": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "avg_pred_matches": 0.0, "false_positive_rate": 0.0}
        res = evaluate_s1_macro(s1_predictions, s1_ground_truth, s1_list=subset)
        fp_rate = 0.0
        if is_zero_match_group:
            fp_count = sum(1 for s1 in subset if len(s1_predictions.get(s1, set())) > 0)
            fp_rate = fp_count / len(subset) * 100.0
        else:
            # False positive rate for non-empty: fraction of predicted matches that are false
            total_pred = res["total_pred_matches"]
            if total_pred > 0:
                total_tp = sum(len(s1_ground_truth.get(s1, set()) & s1_predictions.get(s1, set())) for s1 in subset)
                fp_rate = (total_pred - total_tp) / total_pred * 100.0

        return {
            "count": len(subset),
            "macro_f05": res["macro_f05"],
            "macro_precision": res["macro_precision"],
            "macro_recall": res["macro_recall"],
            "avg_pred_matches": res["avg_pred_matches"],
            "false_positive_rate": fp_rate,
        }

    return {
        "group1_singleton": _eval_subset(group1_s1, is_zero_match_group=False),
        "group2_zero_match": _eval_subset(group2_s1, is_zero_match_group=True),
        "group3_multi_match": _eval_subset(group3_s1, is_zero_match_group=False),
    }


def compute_pair_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes pair-level diagnostic classification metrics.
    Note: These are diagnostic only; the official metric is S1-level macro F0.5.
    """
    y_pred = (y_prob >= threshold).astype(int)
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    loss = float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)))
    brier = float(brier_score_loss(y_true, y_prob))

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = 0.25 * prec + rec
    f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": loss,
        "brier_score": brier,
        "pair_precision": prec,
        "pair_recall": rec,
        "pair_f05": f05,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
