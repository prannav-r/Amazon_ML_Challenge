def compute_f05_macro(ground_truth_dict, predictions_dict):
    """
    ground_truth_dict: {s1_id: set of true matched_ids} (empty set for singletons)
    predictions_dict: {s1_id: set of predicted matched_ids} (empty set for singletons)
    """
    scores = []
    for s1_id, true_set in ground_truth_dict.items():
        pred_set = predictions_dict.get(s1_id, set())
        
        # Singleton entity in ground truth
        if not true_set:
            if not pred_set:
                scores.append(1.0)
            else:
                scores.append(0.0)
            continue
            
        # Non-singleton entity
        if not pred_set:
            scores.append(0.0)
            continue
            
        tp = len(pred_set & true_set)
        if tp == 0:
            scores.append(0.0)
            continue
            
        precision = tp / len(pred_set)
        recall = tp / len(true_set)
        
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        scores.append(f05)
        
    return sum(scores) / len(scores) if scores else 0.0

# Verify with PDF example
gt_ex = {"S1-00001": {"S2-00047", "S3-00812"}}
pred_ex = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
score = compute_f05_macro(gt_ex, pred_ex)
print(f"PDF Example Score: {score:.3f} (Expected: 0.714)")
assert round(score, 3) == 0.714, f"Mismatch: {score}"
print("Metric calculation verified successfully against PDF specification!")
