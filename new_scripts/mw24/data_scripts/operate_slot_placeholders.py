
import os
import json
import argparse
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser(description='Modify the LLM input file for open ended generation.')
    parser.add_argument('--input_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/NecessaryFiles/LLM_input_U_PY_TY_agent_added_n_limited_nn.txt', help='input file')
    # parser.add_argument('--output_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/sandbox/LLM_input_sentences_rem_NA.txt', help='output file')
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    input_file = args.input_file
    # output_file = args.output_file
    output_file = input_file.replace(".txt", "_rem_NA.txt")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    processed_lines = []

    for line in tqdm(lines):
        read_line = json.loads(line)
        new_line = ''
        # print(read_line)
        # print("------")
        # print(type(read_line))
        context_lines_split = read_line[0].split("\n")
        # Go through each examples in the context
        for i, context_line in enumerate(context_lines_split):
            append_slots_str = ''
            # print(context_line)
            # print("-------")
            prefix_original = context_line.split("Slots: ")[0]
            slot_dicts_str = context_line.split("Slots: ")[1]
            # except jsondecodeerror
            # print(prefix_original)
            # print("=====")
            # print(slot_dicts_str)

            try:
                slot_dicts_original = json.loads(slot_dicts_str)
                new_slot_dict = {}
                for domain in slot_dicts_original.keys():
                    # going inside slots of each domain
                    # print(slot_dicts_original[domain])
                    each_domain_slots_new = {}
                    each_domain_slots_original = slot_dicts_original[domain]
                    for slot_key, slot_val in each_domain_slots_original.items():
                        # print(slot_key)
                        # print(slot_val)
                        if slot_val == 'N.A.':
                            # print("NA found")
                            # print(slot_key)
                            continue
                        each_domain_slots_new[slot_key] = slot_val
                    # Add the domain to the new slot dict
                    # new_slot_dict = {domain: each_domain_slots_new}
                    # append to the new slot dict
                    new_slot_dict[domain] = each_domain_slots_new
                append_slots_str = json.dumps(new_slot_dict) + "\n"
                # print(slot_dicts_original)
                # print(append_slots_str)
            except json.decoder.JSONDecodeError:
                append_slots_str = slot_dicts_str
                # print("The last line is not a valid json")
                # print(slot_dicts_str)
        
            one_utterance_with_slots = prefix_original + "Slots: " + append_slots_str
            # print(one_utterance_with_slots)
            # new_line.append(one_utterance_with_slots)
            new_line += one_utterance_with_slots 
            # print("====")
            # breakpoint()
        processed_lines.append(json.dumps([new_line]))

                # breakpoint()
        # print(new_line)
        # print("====")
        # breakpoint()
        # write to the output file
    with open(output_file, 'w') as f:
        f.write("\n".join(processed_lines) + "\n")

if __name__ == '__main__':
    main()
    

