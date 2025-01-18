import os
import json
import argparse
import numpy as np
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser(description='Sort and shuffle slot keys.')
    parser.add_argument('--input_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/NecessaryFiles/LLM_input_U_PY_TY_agent_added_n_limited_nn.txt', help='input file')
    parser.add_argument('--seed', type=str, default='42', help='Seed for shuffling slot keys. Use "Y" for alphabetical sorting. Use "omit" to omit slot keys with "not mentioned" value.')
    parser.add_argument('--output_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/NecessaryFiles/LLM_input_U_PY_TY_agent_added_n_limited_nn_sorted_shuffled.txt', help='output file')
    args = parser.parse_args()
    return args

def sort_and_shuffle_slot_keys(input_file, seed, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    if seed != 'Y' and seed != 'omit':
        np.random.seed(int(seed))
    processed_lines = []

    for line in tqdm(lines):
        read_line = json.loads(line)
        new_line = ''
        context_lines_split = read_line[0].split("\n")

        for i, context_line in enumerate(context_lines_split):
            append_slots_str = ''
            prefix_original = context_line.split("Slots: ")[0]
            slot_dicts_str = context_line.split("Slots: ")[1]

            try:
                slot_dicts_original = json.loads(slot_dicts_str)
                new_slot_dict = {}
                for domain in slot_dicts_original.keys():
                    each_domain_slots_original = slot_dicts_original[domain]

                    # Sort, shuffle, or omit the slot keys
                    slot_keys = list(each_domain_slots_original.keys())
                    if seed == 'Y':
                        slot_keys.sort()
                    elif seed == 'omit':
                        slot_keys = [key for key in slot_keys if each_domain_slots_original[key] != 'not mentioned']
                    else:
                        np.random.shuffle(slot_keys)
                    sorted_or_shuffled_slots = {key: each_domain_slots_original[key] for key in slot_keys}

                    # Add the sorted, shuffled, or omitted slots back to the new slot dict
                    new_slot_dict[domain] = sorted_or_shuffled_slots

                append_slots_str = json.dumps(new_slot_dict) + "\n"
            except json.decoder.JSONDecodeError:
                append_slots_str = slot_dicts_str

            one_utterance_with_slots = prefix_original + "Slots: " + append_slots_str
            new_line += one_utterance_with_slots 

        processed_lines.append(json.dumps([new_line]))

    with open(output_file, 'w') as f:
        f.write("\n".join(processed_lines) + "\n")

    return output_file

if __name__ == '__main__':
    args = arg_parse()
    sort_and_shuffle_slot_keys(args.input_file, args.seed, args.output_file)
