import re
import json
from tqdm import tqdm
import os
import argparse
import numpy as np
import torch
a = torch.rand(1, 1).cuda()
import string
from TODx import remove_punctuations, prepare_data_for_sentence_embedding, calculate_save_sentence_embedding, json_to_tsv_format
from TODx import get_text_dom_in_template1, get_text_dom_in_template2, get_text_dom_slots_in_template1, get_text_dom_slots_in_template2, get_instruction_string 
from TODx import finding_test_labels, find_x_similar_sentences, get_in_simple_user_agent_format, naming_function, get_text_dom_slots_in_template3
from TODx_result_parse import parse_to_results_tsv
import platform


device = torch.device('cuda')

# write to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Domain and Slot Prediction using LLM')
    parser.add_argument('--model', type=str, default='allenai/OLMo-7B-Instruct', help='Model name')
    parser.add_argument('--data_path', type=str, default='/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MULTIWOZ2.4/',  help='Path to the data directory')
    parser.add_argument('--out_dir', type=str, default='/mnt/matylda4/hegde/int_ent/LLM_dialog_state/model_results', help='Path to the LLM output directory')
    parser.add_argument('--dialog_side', default='U', type=str, help='Calculate sentence embedding option: U - for user only sentences, UA - for user-agent sentences')
    parser.add_argument('--punct', default='punct', type=str, help='Remove punctuations from the sentences.: punct - with punctuations, no_punct - without punctuations')   
    parser.add_argument('--context_number', type=int, default=10, help='Number of similar sentences to find in the training set')
    parser.add_argument('--predict', type=str, default='domain', help='Predict domain or slots: domain - predict domain, domain_slots - predict slots')
    parser.add_argument('--input_template', type=int, default=1, help='Input format for LLM: 1, 2, 3... Check the format readme file')
    parser.add_argument('--instruction_preset', type=int, default=1, help='Instruction template for LLM: 0, 1 Check the format readme file')
    parser.add_argument('--skip_computing_embeddings', default=False, help='Skip computing embeddings and TSV files')
    parser.add_argument('--offset_start', type=int, default=0, help='Start offset for the test file')
    parser.add_argument('--offset_add', type=int, default=8000, help='Add offset for the test file')
    parser.add_argument('--lowercase_input', default=False, help='Complete lowercase input to the LLM')
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # output folder path
    model_name, punct_name, predicting = naming_function(args.model, args.punct, args.predict)
    dataset_abbrv = 'mw24'  # change that name of the dataset.
    if args.lowercase_input:
        lower_text = '_lower'
    else:
        lower_text = ''

    op_folder_name = 'TOD1_' + dataset_abbrv + 'Domain_baseline'
    output_folder_path = os.path.join(args.out_dir, op_folder_name)
    os.makedirs(output_folder_path, exist_ok=True)

    # output config file
    config_file = os.path.join(output_folder_path, op_folder_name + '_config.json')
    with open(config_file, 'w') as config_file:
        config_file.write(json.dumps(vars(args), indent=4))
        # Add details of the run
        config_file.write(platform.node())
        config_file.write('\n')

    # Path to the input data file
    data_path = args.data_path
    input_test_data_path = os.path.join(data_path, 'mw24_DST_test.json')
    input_valid_data_path = os.path.join(data_path, 'mw24_DST_valid.json')
    input_train_data_path = os.path.join(data_path, 'mw24_DST_train.json')

    train_tsv_file_path = ''
    test_tsv_file_path = ''
    train_numpy_file_path = ''
    test_numpy_file_path = ''

    # write the labels of train and test set to a TSV file. 
    if args.dialog_side == 'U':
        if args.punct == 'no_punct':
            test_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_test_U_no_punct_turns.tsv')
        elif args.punct == 'punct':
            test_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_test_U_punct_turns.tsv')

        train_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_train_U_turns.tsv')

        if args.skip_computing_embeddings: 
            test_numpy_file_path = os.path.join(output_folder_path, 'mw24_DST_test_U_punct_turns.npy')
            train_numpy_file_path = os.path.join(output_folder_path, 'mw24_DST_train_U_turns.npy')
            if args.punct == 'no_punct':
                test_numpy_file_path = os.path.join(output_folder_path, 'mw24_DST_test_U_no_punct_turns.npy')
        else:
            json_to_tsv_format(input_train_data_path, train_tsv_file_path, args.dialog_side)
            json_to_tsv_format(input_test_data_path, test_tsv_file_path, args.dialog_side)
            train_numpy_file_path = calculate_save_sentence_embedding(train_tsv_file_path, args.dialog_side, punct='punct')
            test_numpy_file_path = calculate_save_sentence_embedding(test_tsv_file_path, args.dialog_side, args.punct)

    elif args.dialog_side == 'UA':
        if args.punct == 'no_punct':
            test_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_test_UA_no_punct_turns.tsv')
        elif args.punct == 'punct':
            test_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_test_UA_punct_turns.tsv')
        train_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_train_UA_turns.tsv')

        if args.skip_computing_embeddings:
            train_numpy_file_path = os.path.join(output_folder_path, 'mw24_DST_train_UA_turns.npy')
            test_numpy_file_path = os.path.join(output_folder_path, 'mw24_DST_test_UA_punct_turns.npy')
            if args.punct == 'no_punct':
                test_numpy_file_path = os.path.join(output_folder_path, 'mw24_DST_test_UA_no_punct_turns.npy')
        else:
            json_to_tsv_format(input_train_data_path, train_tsv_file_path, args.dialog_side)
            json_to_tsv_format(input_test_data_path, test_tsv_file_path, args.dialog_side)
            train_numpy_file_path = calculate_save_sentence_embedding(train_tsv_file_path, args.dialog_side, punct='punct')
            test_numpy_file_path = calculate_save_sentence_embedding(test_tsv_file_path, args.dialog_side, args.punct)

    # Load test numpy file.
    train_embeddings = np.load(train_numpy_file_path)
    test_embeddings = np.load(test_numpy_file_path)
    print("numpy loaded")

    #exit(0)

    # Load test file
    with open(test_tsv_file_path, 'r') as tst_tsv_file:
        test_data = tst_tsv_file.readlines()


    dom2slots = {
    'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'], 
    'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'], 
    'attraction': ['type', 'name', 'area'], 
    'train': ['people', 'leaveAt', 'destination', 'day', 'arriveBy', 'departure'], 
    'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
}

    print("entering loop")
    output_file = os.path.join(output_folder_path, op_folder_name +'_baseline.tsv')
    # output_file = os.path.join(output_folder_path, op_folder_name +'_results.tsv')

    for i, (test_line, test_embedding) in enumerate(zip(test_data, test_embeddings)):
        

        decoded_response = ''
        # find all gold test labels: test_sentence, domain, slots
        gold_fname, gold_sentence, gold_domain, gold_slots = finding_test_labels(test_line)

        # find 20 similar sentences in the train set for each test sentence
        similar_sentences_lines = find_x_similar_sentences(train_embeddings, test_embedding, args.context_number, train_tsv_file_path)

        total_input_to_LLM = ''
        testing_sentence = ''
        
        instruction_string = get_instruction_string(args.instruction_preset, args.predict)

        # Find Domains in the similar sentences which has the highest frequency
        domain_dict = {}
        for sim_sent in similar_sentences_lines:
            _, _, domain, _ = finding_test_labels(sim_sent)
            if domain in domain_dict:
                domain_dict[domain] += 1
            else:
                domain_dict[domain] = 1
        prediction_domain = max(domain_dict, key=domain_dict.get)
        with open(output_file, 'a', encoding='utf-8') as op_file:
            op_file.write(gold_fname + '\t' + gold_domain + '\t' + prediction_domain + '\n')
        continue

if __name__ == "__main__":
    main()
