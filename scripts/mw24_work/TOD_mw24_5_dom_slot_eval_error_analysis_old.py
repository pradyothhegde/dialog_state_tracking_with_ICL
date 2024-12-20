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
        default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/O7BI/Domain/TOD_mw24_O7BI_SD_I1_UA_10_PN_TY/',
        # default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/z_older_experiments/NA_removed_domain_wise_M7BI03/',
        help='Path to the directory containing files',
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
        domain_slot_key = f"{domain}_{slot}"  # Track domain and slot
        if domain_slot_key not in domain_slot_stats:
            domain_slot_stats[domain_slot_key] = {
                'correct': 0,
                'total': 0,
                'NA_correct': 0,
                'NA_total': 0,
                'non_NA_correct': 0,
                'non_NA_total': 0
            }

        if slot in gold_slots:
            total_count += 1
            domain_slot_stats[domain_slot_key]['total'] += 1
            gold_value = gold_slots[slot]

            if gold_value == "N.A.":
                try:
                    predicted_slot_val = TODx.remove_punctuations(str(predicted_slots[slot]))[0]
                except (KeyError, IndexError):
                    predicted_slot_val = ""
                predicted_slot_val = predicted_slot_val.replace(' ', '')
                total_NA_count += 1
                domain_slot_stats[domain_slot_key]['NA_total'] += 1
                if slot not in predicted_slots or predicted_slots == "":
                    correct_count += 1
                    correct_NA_count += 1
                    domain_slot_stats[domain_slot_key]['NA_correct'] += 1
                    domain_slot_stats[domain_slot_key]['correct'] += 1
            else:
                domain_slot_stats[domain_slot_key]['non_NA_total'] += 1
                gold_val = TODx.remove_punctuations(str(gold_value))[0]
                try:
                    predicted_slot_val = TODx.remove_punctuations(str(predicted_slots[slot]))[0]
                except (KeyError, IndexError):
                    predicted_slot_val = ""
                gold_val = gold_val.replace(' ', '')
                predicted_slot_val = predicted_slot_val.replace(' ', '')
                # print(f"Gold: {gold_val}, Predicted: {predicted_slot_val}")
                # breakpoint()
                if slot in predicted_slots and gold_val == predicted_slot_val:
                    correct_count += 1
                    domain_slot_stats[domain_slot_key]['non_NA_correct'] += 1
                    domain_slot_stats[domain_slot_key]['correct'] += 1

    return correct_count, total_count, total_NA_count, correct_NA_count, domain_slot_stats

def main():
    args = arg_parse()
    eval_file = os.path.join(args.eval_folder_path, 'log_files/')

    dom2slots = {
        'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'],
        'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'],
        'attraction': ['type', 'name', 'area'],
        'train': ['people', 'leaveAt', 'destination', 'day', 'arriveBy', 'departure'],
        'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
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
                predicted_output = json_repair.loads(line_dict['processed_output'].strip())     # processed_output or predicted_output

                for g_domain, g_slots in zip(gold_domain, gold_slots):
                    if g_domain in predicted_output:
                        predicted_slots = predicted_output[g_domain]
                        valid_slots = dom2slots.get(g_domain)
                        correct, total, NA_count, correct_NA_count, domain_slot_accuracy = compare_slots(
                            g_slots, predicted_slots, valid_slots, g_domain
                        )
                        total_correct += correct
                        total_slots += total
                        total_NA_count += NA_count
                        total_correct_NA_count += correct_NA_count

                        # Update domain-wise statistics
                        domain_wise_stats[g_domain]['correct'] += correct
                        domain_wise_stats[g_domain]['total'] += total
                        domain_wise_stats[g_domain]['NA_correct'] += correct_NA_count
                        domain_wise_stats[g_domain]['NA_total'] += NA_count
                        domain_wise_stats[g_domain]['non_NA_correct'] += correct - correct_NA_count
                        domain_wise_stats[g_domain]['non_NA_total'] += total - NA_count

                        # Merge domain-slot-wise statistics
                        for key, stats in domain_slot_accuracy.items():
                            if key not in domain_slot_stats:
                                domain_slot_stats[key] = {
                                    'correct': 0,
                                    'total': 0,
                                    'NA_correct': 0,
                                    'NA_total': 0,
                                    'non_NA_correct': 0,
                                    'non_NA_total': 0
                                }
                            domain_slot_stats[key]['correct'] += stats['correct']
                            domain_slot_stats[key]['total'] += stats['total']
                            domain_slot_stats[key]['NA_correct'] += stats['NA_correct']
                            domain_slot_stats[key]['NA_total'] += stats['NA_total']
                            domain_slot_stats[key]['non_NA_correct'] += stats['non_NA_correct']
                            domain_slot_stats[key]['non_NA_total'] += stats['non_NA_total']

    # Print evaluation summary
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
    print(f"Total Non NA Slots: {Total_non_NA_slots}")
    Total_correct_non_NA_slots = total_correct - total_correct_NA_count
    Non_na_accuracy = (Total_correct_non_NA_slots / Total_non_NA_slots) * 100 if Total_non_NA_slots > 0 else 0
    print(f"Non NA Accuracy: {Non_na_accuracy:.2f}%")
    print("===============")

    # print("Domain-Wise Slot Accuracy:")
    # for domain, stats in domain_wise_stats.items():
    #     domain_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
    #     domain_NA_accuracy = (stats['NA_correct'] / stats['NA_total']) * 100 if stats['NA_total'] > 0 else 0
    #     domain_non_NA_accuracy = (stats['non_NA_correct'] / stats['non_NA_total']) * 100 if stats['non_NA_total'] > 0 else 0
    #     print(f"  {domain}: Overall Accuracy: {domain_accuracy:.2f}%")
    #     print(f"    NA Accuracy: {domain_NA_accuracy:.2f}%")
    #     print(f"    Non-NA Accuracy: {domain_non_NA_accuracy:.2f}%")

    # print("===============")
    # print("Domain-Slot-Wise Accuracy:")
    # for domain_slot, stats in domain_slot_stats.items():
    #     slot_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
    #     NA_accuracy = (stats['NA_correct'] / stats['NA_total']) * 100 if stats['NA_total'] > 0 else 0
    #     non_NA_accuracy = (stats['non_NA_correct'] / stats['non_NA_total']) * 100 if stats['non_NA_total'] > 0 else 0
    #     print(f"  {domain_slot}:")
    #     print(f"    Overall Accuracy: {slot_accuracy:.2f}%")
    #     print(f"    NA Accuracy: {NA_accuracy:.2f}% (Occurrences: {stats['NA_total']})")
    #     print(f"    Non-NA Accuracy: {non_NA_accuracy:.2f}% (Occurrences: {stats['non_NA_total']})")

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
