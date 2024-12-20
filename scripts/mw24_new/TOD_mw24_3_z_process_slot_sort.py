import os
import json
import argparse
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser(description='Modify the LLM input file for open ended generation.')
    parser.add_argument('--input_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/NecessaryFiles/LLM_input_sentences_U.txt', help='input file')
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    input_file = args.input_file
    output_file = input_file.replace(".txt", "_sorted.txt")

    with open(input_file, 'r') as f:
        lines = f.readlines()

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

                    # Sort the slots by keys
                    sorted_slots = dict(sorted(each_domain_slots_original.items()))

                    # Add the sorted slots back to the new slot dict
                    new_slot_dict[domain] = sorted_slots

                append_slots_str = json.dumps(new_slot_dict) + "\n"
            except json.decoder.JSONDecodeError:
                append_slots_str = slot_dicts_str

            one_utterance_with_slots = prefix_original + "Slots: " + append_slots_str
            new_line += one_utterance_with_slots 

        processed_lines.append(json.dumps([new_line]))

    with open(output_file, 'w') as f:
        f.write("\n".join(processed_lines) + "\n")

if __name__ == '__main__':
    main()
