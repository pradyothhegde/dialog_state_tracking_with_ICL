import numpy as np
import ast
import argparse
import json
import os
import debug_TODx as TODx
import json_repair


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Evaluate open-ended domain-wise slot prediction.'
    )
    parser.add_argument(
        '--input_file_path',
        type=str,
        default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/TOD_mw24_O7BI_U_T_P_10_sltW/log_files/',
        help='Path to the directory containing input JSON files',
    )
    args = parser.parse_args()
    return args


def compare_slots(gold_slots, predicted_slots, domain_slots, slot_stats, domain):
    correct_count = 0
    total_count = 0
    total_NA_count = 0
    correct_NA_count = 0

    for slot in domain_slots:
        domain_slot_key = f"{domain}_{slot}"
        slot_stats[domain_slot_key]['total'] += 1

        if slot in gold_slots:
            total_count += 1
            gold_value = TODx.remove_punctuations([gold_slots[slot]])[0]
            predicted_value = TODx.remove_punctuations([predicted_slots.get(slot, "").strip()])[0]

            if domain == "hotel" and slot == "type":
                print(f"Gold: {gold_value}, Predicted: {predicted_value}")

            if gold_value == predicted_value:
                correct_count += 1
                slot_stats[domain_slot_key]['correct'] += 1

            if gold_value == "na":
                total_NA_count += 1
                if gold_value == predicted_value:
                    correct_NA_count += 1

    return correct_count, total_count, total_NA_count, correct_NA_count


def main():
    args = arg_parse()
    eval_file = args.input_file_path
    print("Evaluation Started")

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
    domain_wise_stats = {domain: {'correct': 0, 'total': 0} for domain in dom2slots}

    # Initialize domain_slot-wise stats
    slot_stats = {f"{domain}_{slot}": {'correct': 0, 'total': 0} for domain, slots in dom2slots.items() for slot in slots}

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
                predicted_output = line_dict['processed_output']
                # predicted_output = predicted_output.strip().replace('\n', ' ').replace('\t', ' ')

                for g_domain, g_slots in zip(gold_domain, gold_slots):
                    if g_domain in predicted_output:
                        predicted_slots = predicted_output[g_domain]
                        predicted_slots = predicted_slots
                        valid_slots = dom2slots.get(g_domain)

                        correct, total, NA_count, correct_NA_count = compare_slots(
                            g_slots, predicted_slots, valid_slots, slot_stats, g_domain
                        )
                        
                        domain_wise_stats[g_domain]['correct'] += correct
                        domain_wise_stats[g_domain]['total'] += total
                        
                        total_correct += correct
                        total_slots += total
                        total_NA_count += NA_count
                        total_correct_NA_count += correct_NA_count

    # Final Evaluation Summary
    print("===============")
    print("Evaluation Summary")
    print(f"Total Correct Slots: {total_correct}")
    print(f"Total Slots: {total_slots}")
    accuracy = (total_correct / total_slots) * 100 if total_slots > 0 else 0
    print(f"Slot Prediction Accuracy: {accuracy:.2f}%")
    print(f"Total NA Slots: {total_NA_count}")
    print(f"Correct NA Slots: {total_correct_NA_count}")
    print(f"NA Accuracy: {(total_correct_NA_count / total_NA_count) * 100:.2f}%")
    print("===============")

    print("Domain-Wise Slot Accuracy:")
    for domain, stats in domain_wise_stats.items():
        domain_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {domain}: {domain_accuracy:.2f}%")

    print("===============")
    print("Domain-Slot-Wise Accuracy:")
    for domain_slot, stats in slot_stats.items():
        slot_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {domain_slot}: Accuracy: {slot_accuracy:.2f}%, Occurrences: {stats['total']}")


if __name__ == '__main__':
    main()


# check for blank domains, which in turn corresponds to empty slots