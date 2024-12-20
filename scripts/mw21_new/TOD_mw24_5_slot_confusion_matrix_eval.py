import numpy as np
import ast
import argparse
import json
import os
import TODx as TODx
import json_repair


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Evaluate open-ended domain-wise slot prediction.'
    )
    parser.add_argument(
        '--eval_folder_path',
        type=str,
        default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/mw24/O7BI/usr_agnt_ualimtd_srtd_TOD_mw24_O7BI_SS_I2_U_10_PY_TY',
        help='Path to the directory containing input JSON files',
    )
    args = parser.parse_args()
    return args


def compare_slots(gold_slots, predicted_slots, domain_slots, domain):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    domain_slot_stats = {}

    for slot in domain_slots:
        domain_slot_key = f"{domain}_{slot}"
        if domain_slot_key not in domain_slot_stats:
            domain_slot_stats[domain_slot_key] = {
                'tp': 0,
                'tn': 0,
                'fp': 0,
                'fn': 0,
            }

        gold_value = TODx.remove_punctuations([gold_slots.get(slot, "na").strip()])[0]
        predicted_value = TODx.remove_punctuations([predicted_slots.get(slot, "na").strip()])[0]

        # Normalize values by removing white spaces
        gold_value = gold_value.replace(" ", "")
        predicted_value = predicted_value.replace(" ", "")

        # print(f"Gold Value: {gold_value}")
        # print(f"predicted Value: {predicted_value}")

        if gold_value == "na":
            if predicted_value in ("na", ""):
                tn += 1
                domain_slot_stats[domain_slot_key]['tn'] += 1
            else:
                fp += 1
                domain_slot_stats[domain_slot_key]['fp'] += 1
        else:
            if gold_value == predicted_value:
                tp += 1
                domain_slot_stats[domain_slot_key]['tp'] += 1
            else:
                fn += 1
                domain_slot_stats[domain_slot_key]['fn'] += 1

    return tp, tn, fp, fn, domain_slot_stats


def main():
    args = arg_parse()
    eval_file = os.path.join(args.eval_folder_path, 'log_files/')
    print("Evaluation Started")

    dom2slots = {
        'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'],
        'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'],
        'attraction': ['type', 'name', 'area'],
        'train': ['people', 'leaveAt', 'destination', 'day', 'arriveBy', 'departure'],
        'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
    }

    overall_tp = 0
    overall_tn = 0
    overall_fp = 0
    overall_fn = 0
    domain_wise_stats = {
        domain: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        for domain in dom2slots
    }
    domain_slot_stats = {}

    for file in os.listdir(eval_file):
        if file.endswith('.json'):
            print(f"Processing file: {file}")
            log_file = os.path.join(eval_file, file)

            with open(log_file, 'r') as f:
                eval_data = f.readlines()

            for line in eval_data:
                line_dict = json.loads(line.strip())
                gold_domain = ast.literal_eval(line_dict['gold_domain'])
                gold_slots = ast.literal_eval(line_dict['gold_slots'])
                predicted_output = json_repair.loads(line_dict['processed_output'])

                for g_domain, g_slots in zip(gold_domain, gold_slots):
                    if g_domain in predicted_output:
                        predicted_slots = predicted_output[g_domain]
                        valid_slots = dom2slots.get(g_domain)

                        tp, tn, fp, fn, slot_stats = compare_slots(
                            g_slots, predicted_slots, valid_slots, g_domain
                        )

                        overall_tp += tp
                        overall_tn += tn
                        overall_fp += fp
                        overall_fn += fn

                        domain_wise_stats[g_domain]['tp'] += tp
                        domain_wise_stats[g_domain]['tn'] += tn
                        domain_wise_stats[g_domain]['fp'] += fp
                        domain_wise_stats[g_domain]['fn'] += fn

                        for key, stats in slot_stats.items():
                            if key not in domain_slot_stats:
                                domain_slot_stats[key] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
                            domain_slot_stats[key]['tp'] += stats['tp']
                            domain_slot_stats[key]['tn'] += stats['tn']
                            domain_slot_stats[key]['fp'] += stats['fp']
                            domain_slot_stats[key]['fn'] += stats['fn']

    # Final Evaluation Summary
    print("===============")
    print("Evaluation Summary")
    print(f"True Positives (TP): {overall_tp}")
    print(f"True Negatives (TN): {overall_tn}")
    print(f"False Positives (FP): {overall_fp}")
    print(f"False Negatives (FN): {overall_fn}")

    precision = (overall_tp / (overall_tp + overall_fp)) * 100 if (overall_tp + overall_fp) > 0 else 0
    recall = (overall_tp / (overall_tp + overall_fn)) * 100 if (overall_tp + overall_fn) > 0 else 0
    # f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    # accuracy
    accuracy = ((overall_tp + overall_tn) / (overall_tp + overall_tn + overall_fp + overall_fn)) * 100

    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"Accuracy: {accuracy:.2f}%")
    print("===============")

    # Save results
    folder_name = args.eval_folder_path.split('/')[-2]
    out_file = os.path.join(args.eval_folder_path, folder_name + '_results.json')
    with open(out_file, 'w') as f:
        json.dump({
            'tp': overall_tp,
            'tn': overall_tn,
            'fp': overall_fp,
            'fn': overall_fn,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'domain_wise_stats': domain_wise_stats,
            'domain_slot_stats': domain_slot_stats
        }, f, indent=4)


if __name__ == '__main__':
    main()
