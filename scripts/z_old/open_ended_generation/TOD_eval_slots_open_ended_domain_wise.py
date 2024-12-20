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
        default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/NA_removed_domain_wise_O7BI/log_files/',
        help='Path to the directory containing input JSON files',
    )
    args = parser.parse_args()
    return args


def compare_slots(gold_slots, predicted_slots, domain_slots):
    correct_count = 0
    total_count = 0
    total_NA_count = 0
    correct_NA_count = 0

    for slot in domain_slots:
        if slot in gold_slots:
            total_count += 1
            gold_value = gold_slots[slot]
            if gold_value == "N.A.":
                total_NA_count += 1
                # Count as correct if slot is absent in predicted_slots
                if slot not in predicted_slots:
                    correct_count += 1
                    correct_NA_count += 1
            else:
                # Count as correct if slot is present and values match
                if slot in predicted_slots and TODx.remove_punctuations(str(gold_value)) == TODx.remove_punctuations(str(predicted_slots[slot])): # if there is slot of same name in prediction, and the value is same
                # if slot in predicted_slots and TODx.remove_punctuations(str(gold_value).replace(" ", "")) == TODx.remove_punctuations(str(predicted_slots[slot]).replace(" ", "")): # if there is slot of same name in prediction, and the value is same
                    correct_count += 1

    return correct_count, total_count, total_NA_count, correct_NA_count


def main():
    args = arg_parse()
    eval_file = args.input_file_path

    dom2slots = {
    'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'], 
    'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'], 
    'attraction': ['type', 'name', 'area'], 
    'train': ['people', 'leaveAt', 'destination', 'day', 'arriveBy', 'departure'], 
    'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
}

    # Overall evaluation counters
    total_correct = 0
    total_slots = 0
    total_NA_count = 0
    total_correct_NA_count = 0

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
                # print(f"Predicted Output: {line_dict['predicted_output']}")
                predicted_output = json_repair.loads(line_dict['processed_output'].strip())


                for g_domain, g_slots in zip(gold_domain, gold_slots):
                    if g_domain in predicted_output:
                        predicted_slots = predicted_output[g_domain]
                        valid_slots = dom2slots.get(g_domain)
                        correct, total, NA_count, correct_NA_count = compare_slots(g_slots, predicted_slots, valid_slots)
                        total_correct += correct
                        total_slots += total
                        total_NA_count += NA_count
                        total_correct_NA_count += correct_NA_count

    print("===============")
    print("Evaluation Summary")
    print(f"Total Correct Slots: {total_correct}")
    print(f"Total Slots: {total_slots}")
    accuracy = (total_correct / total_slots) * 100 if total_slots > 0 else 0
    print(f"Slot Prediction Accuracy: {accuracy:.2f}%")
    print(f"Total NA Slots: {total_NA_count}")
    print(f"correct_NA_count: {total_correct_NA_count}")
    print(f"NA Accuracy: {(total_correct_NA_count / total_NA_count) * 100:.2f}%")
    print("===============")
    print("Non NA Slots")
    print(f"Total Non NA Slots: {total_slots - total_NA_count}")
    print(f"Non NA accuacy: {(total_correct-total_correct_NA_count)/(total_slots-total_NA_count)}")


if __name__ == '__main__':
    main()







# slot accuracy for each domain
# accuracy for NA and Non-NA slots
# Confusion matrix?


# accumulating slots for per slot prediction? 
