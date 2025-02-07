# To read the input file to LLM. This input file would contain the context and test utterance. 
# The two dictionaries pointed in the arrow are the ground truth labels and the slots extracted from the context example. 
# Use them for error analysis.

import json
import os
import argparse
import ast

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MW24/demo/MW24_MP_ST_PH-nm_SO_Labse_NN-3_UA_SV/MW24_MP_ST_PH-nm_SO_Labse_NN-3_UA_SV.txt')
    parser.add_argument('--ground_truth_test_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MW24/punct/MW24_MP_NST_PH-nm_SO_Labse_NN-10_UA_SV/MW24_test_UA_MP.tsv')
    return parser.parse_args()

def finding_test_labels(test_line):
    test_line_data = test_line.strip().split('\t')
    gold_filename = test_line_data[0]
    gold_sentence = test_line_data[1]
    gold_sentence = gold_sentence.replace('---', ' ')
    gold_domain = test_line_data[2]
    gold_slots = test_line_data[3]
    return gold_filename, gold_sentence, gold_domain, gold_slots

def get_active_context_slots():
    pass

def main():
    args = parse_args()
    file = args.context_file
    with open(file, 'r') as f:
        data = f.readlines()
    with open(args.ground_truth_test_file, 'r') as f:
        ground_truth_data = f.readlines()
    for line, test_line in zip(data, ground_truth_data):
        _, _, gold_domain, gold_slots = finding_test_labels(test_line)
        # convert to dictionary of domain: {slots_keys: slots_values}
        gold_slots_dict = {}
        for gold_dom, gold_slo in zip(ast.literal_eval(gold_domain), ast.literal_eval(gold_slots)):
            gold_slots_dict[gold_dom] = gold_slo
        print(gold_slots_dict) # <--------------------

        line_string = ast.literal_eval(line)[0]
        # print(line_string)
        # consider the string from "Slots:" to the end of the line
        line_string_list = line_string.split('\n')
        for line_string in line_string_list:            
            try:
                slots_dictionary = ast.literal_eval(line_string[line_string.find("Slots:")+6:].strip())
                print("-------------------")
                print(slots_dictionary) # <--------------------
                # {'taxi': {'arriveBy': 'not mentioned', 'departure': 'sidney sussex college', 'destination': 'pizza hut fenditton', 'leaveAt': 'not mentioned'}}
                breakpoint()
            except:
                continue


if __name__ == '__main__':
    main()