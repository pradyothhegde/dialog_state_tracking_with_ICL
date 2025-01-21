import os
import json
import argparse
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser(description='Modify the LLM input file for open ended generation.')
    parser.add_argument('--input_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/NecessaryFiles/LLM_input_U_PY_TY_agent_added_n_limited_nn.txt', help='input file')
    parser.add_argument('--slot_placeholder', type=str, default='not mentioned', help='Slot placeholder to use if not "not mentioned"')
    parser.add_argument('--output_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/NecessaryFiles/LLM_input_U_PY_TY_agent_added_n_limited_nn_rem_NA.txt', help='output file')
    args = parser.parse_args()
    return args

def operate_slot_placeholders(input_file, slot_placeholder, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    processed_lines = []

    for line in tqdm(lines):
        read_line = json.loads(line)
        new_line = ''
        context_lines_split = read_line[0].split("\n")
        # Go through each examples in the context
        for i, context_line in enumerate(context_lines_split):
            append_slots_str = ''
            prefix_original = context_line.split("Slots: ")[0]
            slot_dicts_str = context_line.split("Slots: ")[1]

            try:
                slot_dicts_original = json.loads(slot_dicts_str)
                new_slot_dict = {}
                for domain in slot_dicts_original.keys():
                    each_domain_slots_new = {}
                    each_domain_slots_original = slot_dicts_original[domain]
                    for slot_key, slot_val in each_domain_slots_original.items():
                        if slot_val == 'not mentioned':
                            if slot_placeholder == 'omit':
                                continue
                            elif slot_placeholder == 'empty':
                                slot_val = ''
                            elif slot_placeholder != 'not mentioned':
                                slot_val = slot_placeholder
                        each_domain_slots_new[slot_key] = slot_val
                    new_slot_dict[domain] = each_domain_slots_new
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
    operate_slot_placeholders(args.input_file, args.slot_placeholder, args.output_file)

