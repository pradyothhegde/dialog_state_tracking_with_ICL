# This script reads the created file and prints each demonstration in each test case and also prints the number of demonstrations in each test case.

import os
import argparse
import ast

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MULTIWOZ2.4/processed_data/MW24_OP_ST_PH-omit_SO_Labse_NN-10_U_SKV/MW24_OP_ST_PH-omit_SO_Labse_NN-10_U_SKV.txt')
    return parser.parse_args()

def main():
    args = parse_args()
    inner_line_count = 0
    file = args.file_path
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_read = ast.literal_eval(line)
            # print(len(line_read))
            # print(line_read)
            for items in line_read:
                print(items)                                        # prints each demonstration in each test case.
                number_of_lines = len(items.split('\n'))
                print(f"Number of lines: {number_of_lines}")        # prints number of demonstrations in each test case.
            breakpoint()


if __name__ == '__main__':
    main()
