import os
import json
import argparse
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser(description='Modify the LLM input file for zero shot open ended generation.')
    parser.add_argument('--input_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/sw/NecessaryFiles/SW_LLM_input_UA.txt', help='input file')
    # parser.add_argument('--output_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/sandbox/LLM_input_sentences_rem_user_tag.txt', help='output file')
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    input_file = args.input_file
    # output_file = args.output_file
    output_file = input_file.replace('.txt', '_TN.txt')

    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    processed_lines = []
    
    for line in tqdm(lines):
        read_line = json.loads(line)[0]

        # remove User: tag
        read_line = read_line.replace('User: ', '')
        read_line = read_line.replace('Agent: ', '')
        
        processed_lines.append(json.dumps([read_line]))

    # write to output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(processed_lines)+'\n')

if __name__ == '__main__':
    main()
    