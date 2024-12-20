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
        default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/sw/sw_results/TOD_sw_O7BI_SS_I2_UA_10_PY_TY/',
        # default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/newer/TOD_mw24_M7BIV03_SS_I2_U_10_PY_TY/',
        help='Path to the directory containing input JSON files',
    )
    args = parser.parse_args()
    return args


def compare_slots(gold_slots, predicted_slots, domain_slots, domain):
    correct_count = 0
    total_count = 0
    total_NA_count = 0
    correct_NA_count = 0
    domain_slot_stats = {}

    for slot in domain_slots:
        domain_slot_key = f"{domain}_{slot}"
        if domain_slot_key not in domain_slot_stats:
            domain_slot_stats[domain_slot_key] = {
                'correct': 0,
                'total': 0,
                'NA_correct': 0,
                'NA_total': 0,
                'non_NA_correct': 0,
                'non_NA_total': 0,
            }

        domain_slot_stats[domain_slot_key]['total'] += 1

        if slot in gold_slots:
            total_count += 1
            gold_value = TODx.remove_punctuations([gold_slots[slot]])[0]
            predicted_value = TODx.remove_punctuations([predicted_slots.get(slot, "").strip()])[0]

            # remove white spaces
            gold_value = gold_value.replace(" ", "")
            predicted_value = predicted_value.replace(" ", "")


            if gold_value == predicted_value:
                correct_count += 1
                domain_slot_stats[domain_slot_key]['correct'] += 1

            if gold_value == "na":
                total_NA_count += 1
                domain_slot_stats[domain_slot_key]['NA_total'] += 1
                if gold_value == predicted_value or predicted_value == "":
                    correct_NA_count += 1
                    domain_slot_stats[domain_slot_key]['NA_correct'] += 1
            else:
                domain_slot_stats[domain_slot_key]['non_NA_total'] += 1
                if gold_value == predicted_value:
                    domain_slot_stats[domain_slot_key]['non_NA_correct'] += 1

    return correct_count, total_count, total_NA_count, correct_NA_count, domain_slot_stats


def main():
    args = arg_parse()
    eval_file = os.path.join(args.eval_folder_path, 'log_files/')
    print("Evaluation Started")

    dom2slots = {
    'attraction': ['area', 'name', 'type'],
    'hospital': ['department'],
    'hotel': ['area', 'day', 'internet', 'name', 'parking', 'people', 'pricerange', 'stars', 'stay', 'type'],
    'profile': ['email', 'idnumber', 'name', 'phonenumber', 'platenumber'],
    'restaurant': ['area', 'day', 'food', 'name', 'people', 'pricerange', 'time'],
    'taxi': ['arriveBy', 'departure', 'destination', 'leaveAt'],
    'train': ['arriveBy', 'day', 'departure', 'destination', 'leaveAt', 'people']
    }

    total_correct = 0
    total_slots = 0
    total_NA_count = 0
    total_correct_NA_count = 0
    domain_wise_stats = {
        domain: {'correct': 0, 'total': 0, 'NA_correct': 0, 'NA_total': 0, 'non_NA_correct': 0, 'non_NA_total': 0}
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

                        correct, total, NA_count, correct_NA_count, slot_stats = compare_slots(
                            g_slots, predicted_slots, valid_slots, g_domain
                        )

                        total_correct += correct
                        total_slots += total
                        total_NA_count += NA_count
                        total_correct_NA_count += correct_NA_count

                        domain_wise_stats[g_domain]['correct'] += correct
                        domain_wise_stats[g_domain]['total'] += total
                        domain_wise_stats[g_domain]['NA_correct'] += correct_NA_count
                        domain_wise_stats[g_domain]['NA_total'] += NA_count
                        domain_wise_stats[g_domain]['non_NA_correct'] += correct - correct_NA_count
                        domain_wise_stats[g_domain]['non_NA_total'] += total - NA_count

                        for key, stats in slot_stats.items():
                            if key not in domain_slot_stats:
                                domain_slot_stats[key] = {
                                    'correct': 0,
                                    'total': 0,
                                    'NA_correct': 0,
                                    'NA_total': 0,
                                    'non_NA_correct': 0,
                                    'non_NA_total': 0,
                                }
                            domain_slot_stats[key]['correct'] += stats['correct']
                            domain_slot_stats[key]['total'] += stats['total']
                            domain_slot_stats[key]['NA_correct'] += stats['NA_correct']
                            domain_slot_stats[key]['NA_total'] += stats['NA_total']
                            domain_slot_stats[key]['non_NA_correct'] += stats['non_NA_correct']
                            domain_slot_stats[key]['non_NA_total'] += stats['non_NA_total']

    # Final Evaluation Summary
    print("===============")
    print("Evaluation Summary")
    print(f"Total Correct Slots: {total_correct}")
    print(f"Total Slots: {total_slots}")
    accuracy = (total_correct / total_slots) * 100 if total_slots > 0 else 0
    print(f"Slot Prediction Accuracy: {accuracy:.2f}%")
    print(f"Total NA Slots: {total_NA_count}")
    print(f"Correct NA Slots: {total_correct_NA_count}")
    NA_accuracy = (total_correct_NA_count / total_NA_count) * 100 if total_NA_count > 0 else 0
    print(f"NA Accuracy: {NA_accuracy:.2f}%")
    Total_non_NA_slots = total_slots - total_NA_count
    Total_correct_non_NA_slots = total_correct - total_correct_NA_count
    Non_na_accuracy = (Total_correct_non_NA_slots / Total_non_NA_slots) * 100 if Total_non_NA_slots > 0 else 0
    print(f"Non NA Accuracy: {Non_na_accuracy:.2f}%")
    print("===============")

    # Save results
    folder_name = args.eval_folder_path.split('/')[-2]
    out_file = os.path.join(args.eval_folder_path, folder_name + '_results.json')
    with open(out_file, 'w') as f:
        json.dump({
            'total_correct': total_correct,
            'total_slots': total_slots,
            'accuracy': accuracy,
            'total_NA_count': total_NA_count,
            'total_correct_NA_count': total_correct_NA_count,
            'NA_accuracy': NA_accuracy,
            'Total_non_NA_slots': Total_non_NA_slots,
            'Total_correct_non_NA_slots': Total_correct_non_NA_slots,
            'Non_na_accuracy': Non_na_accuracy,
            'domain_wise_stats': domain_wise_stats,
            'domain_slot_stats': domain_slot_stats
        }, f, indent=4)


if __name__ == '__main__':
    main()
