import os
import json
import argparse
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate domain')
    parser.add_argument('--input_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/experiments/MW24/domain/O7BI_MW24_MP_ST_PH-nm_SO_Labse_NN-10_UA_SV.tsv', help='Input file')
    return parser.parse_args()

def calculate_precision_recall(predictions, ground_truths):
    """Calculates precision and recall for domain prediction.

    Args:
        predictions: A list of predicted domain lists.
        ground_truths: A list of ground truth domain lists.

    Returns:
        A tuple of (precision, recall, f1). Returns (0, 0, 0) if there are no
        ground truths or predictions.
    """
    
    true_positives = 0
    predicted_positives = 0
    actual_positives = 0

    for prediction, ground_truth in zip(predictions, ground_truths):
        
        # Convert to sets for easier comparison
        prediction_set = set(prediction)
        ground_truth_set = set(ground_truth)

        true_positives += len(prediction_set.intersection(ground_truth_set))    # TP
        predicted_positives += len(prediction_set)  # TP + FP
        actual_positives += len(ground_truth_set)   # TP + FN

    if predicted_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / predicted_positives    # TP / (TP + FP)

    if actual_positives == 0:
        recall = 0.0
    else:
        recall = true_positives / actual_positives  # TP / (TP + FN)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)    # 2 * (P * R) / (P + R)

    return precision, recall, f1

def calculate_domain_accuracy(predictions, ground_truths):
    """Calculates domain accuracy.

    Args:
        predictions: A list of predicted domain lists.
        ground_truths: A list of ground truth domain lists.

    Returns:
        The domain accuracy (float).
    """
    correct_predictions = 0
    total_predictions = len(predictions)

    for prediction, ground_truth in zip(predictions, ground_truths):
        # Convert to sets for comparison
        prediction_set = set(prediction)
        ground_truth_set = set(ground_truth)

        if prediction_set == ground_truth_set:
            correct_predictions += 1

    if total_predictions == 0:
        return 0.0
    else:
        return correct_predictions / total_predictions

def main():
    args = parse_arguments()

    ground_truth_domains = []
    predicted_domains = []

    with open(args.input_file, 'r') as f:
        for line in f:
            filename, gt_str, pred_str = line.strip().split('\t')
            
            # Parse the string representations of lists into actual lists
            try:
                ground_truth = json.loads(gt_str.replace("'", "\""))  # Replace single quotes with double quotes for valid JSON
                prediction = json.loads(pred_str.replace("'", "\""))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for line: {line.strip()}")
                print(f"Error details: {e}")
                continue
            
            ground_truth_domains.append(ground_truth)
            predicted_domains.append(prediction)
    
    precision, recall, f1 = calculate_precision_recall(predicted_domains, ground_truth_domains)
    accuracy = calculate_domain_accuracy(predicted_domains, ground_truth_domains)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Domain Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()