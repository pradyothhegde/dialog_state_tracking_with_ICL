import os
import json
from tqdm import tqdm

def remove_speaker_tags(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    processed_lines = []
    
    for line in tqdm(lines):
        read_line = json.loads(line)[0]

        # remove User: and Agent: tags
        read_line = read_line.replace('User: ', '')
        read_line = read_line.replace('Agent: ', '')
        
        processed_lines.append(json.dumps([read_line]))

    # write to output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(processed_lines) + '\n')

    return output_file

def add_dialog_at_the_start(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    processed_lines = []
    
    for line in tqdm(lines):
        read_line = json.loads(line)[0]

        # add dialog: at the start of the line
        read_line = 'dialog: ' + read_line
        
        processed_lines.append(json.dumps([read_line]))

    # write to output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(processed_lines) + '\n')

    return output_file

if __name__ == '__main__':
    import argparse

    def arg_parse():
        parser = argparse.ArgumentParser(description='Modify the LLM input file for zero shot open ended generation.')
        parser.add_argument('--input_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/NecessaryFiles/LLM_input_sentences_UA_PN.txt', help='input file')
        parser.add_argument('--output_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/NecessaryFiles/LLM_input_sentences_UA_PN_TN.txt', help='output file')
        args = parser.parse_args()
        return args

    args = arg_parse()
    remove_speaker_tags(args.input_file, args.output_file)