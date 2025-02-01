# export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1); export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_EVALUATE_OFFLINE=1; /homes/kazi/hegde/miniconda3/envs/dialog/bin/python /mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/new_scripts/mw24/DST_mw24_3_create_files.py

import os
import json
from time import sleep
import tqdm
import ast
import argparse
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from TODx import finding_test_labels, json_to_tsv_format, calculate_save_sentence_embedding, remove_punctuations
from data_scripts.get_out_file_name import get_out_file_name
from data_scripts.create_TSV_from_json import create_TSV_from_json
from data_scripts.compute_save_sentence_embedding import compute_save_sentence_embedding
from data_scripts.check_context_token_limit import check_context_token_limit
from data_scripts.remove_speaker_tags import remove_speaker_tags
from data_scripts.operate_slot_placeholders import operate_slot_placeholders  # Import the function
from data_scripts.sort_slot_keys import sort_and_shuffle_slot_keys  # Import the function


# arguments 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder_path", type=str, default="/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MULTIWOZ2.4")
    parser.add_argument("--output_folder_path", type=str, default="/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MW24/test")
    parser.add_argument("--dataset", type=str, default="MW24")
    parser.add_argument("--LLM_model_tokenizer", type=str, default="allenai/OLMo-7B-Instruct", help="allenai/OLMo-7B-Instruct | ")
    parser.add_argument("--LLM_model_tokenizer_limit", type=int, default=2048, help="Maximum window length of tokenizer - 2048 | ")

    parser.add_argument("--punct", type=str, default="O", help="Original - O | No punctuation - N | Model punctuation - M")
    parser.add_argument("--speaker_tag", type=str, default="Y", help="Y or N")
    parser.add_argument("--slot_placeholder", type=str, default="not mentioned", help="not mentioned | N.A. | empty("") | omit") 
    parser.add_argument("--slot_key_sort", type=str, default="Y", help="Y | N | seed number")

    parser.add_argument("--sentence_embedding_model", type=str, default="sentence-transformers/LaBSE", help="sentence-transformers/LaBSE - Labse | sergioburdisso/dialog2flow-single-bert-base - D2F ")
    parser.add_argument("--NNcount", type=int, default=10, help="number of nearest neighbours to consider - 3 | 5 | 10")

    parser.add_argument("--dialog_history", type=str, default="UA", help="user -U | user agent - UA")

    parser.add_argument("--decoding", type=str, default="SV", help="slot key and value given domain (SKV) | slot value given slot key (SV)")

    args = parser.parse_args()
    return args


def main():
    # arguments
    args = parse_args()
    NNcount = args.NNcount
    dialog_side = args.dialog_history
    print("entered program")

    # get the output file name
    output_file_name = get_out_file_name(args)
    output_folder_path = os.path.join(args.output_folder_path, output_file_name)
    output_file_name = os.path.join(output_folder_path, f"{output_file_name}.txt")
    print(output_folder_path)

    # if the output file exists, print and exit the program
    if os.path.exists(output_file_name):
        print(f"Output file already exists: {output_file_name}")
        return

    # if output path does not exists, create it
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # breakpoint()

    # input file paths
    input_test_data_path = os.path.join(args.data_folder_path, 'mw24_DST_test.json')
    input_train_data_path = os.path.join(args.data_folder_path, 'mw24_DST_train.json')
    input_val_data_path = os.path.join(args.data_folder_path, 'mw24_DST_valid.json')

    # TSV file names
    test_TSV_file_path = os.path.join(output_folder_path, f"MW24_test_{args.dialog_history}_{args.punct}P.tsv")
    train_TSV_file_path = os.path.join(output_folder_path, f"MW24_train_{args.dialog_history}_{args.punct}P.tsv")
    
    # create TSV files      # The punctuations are dealt here.
    create_TSV_from_json(input_test_data_path, test_TSV_file_path, args)
    create_TSV_from_json(input_train_data_path, train_TSV_file_path, args)

    # exit()

    # create numpy files
    test_numpy_file_path = compute_save_sentence_embedding(test_TSV_file_path)
    train_numpy_file_path = compute_save_sentence_embedding(train_TSV_file_path)

    # Temporary paths
    # test_numpy_file_path = '/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MULTIWOZ2.4/processed_data/MW24_OP_ST_PH-nm_SO_Labse_NN-10_U_SKV/MW24_test_U_O.npy'
    # train_numpy_file_path = '/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MULTIWOZ2.4/processed_data/MW24_OP_ST_PH-nm_SO_Labse_NN-10_U_SKV/MW24_train_U_O.npy'
    # test_TSV_file_path = '/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MULTIWOZ2.4/processed_data/MW24_OP_ST_PH-nm_SO_Labse_NN-10_U_SKV/MW24_test_U_O.tsv'
    # train_TSV_file_path = '/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MULTIWOZ2.4/processed_data/MW24_OP_ST_PH-nm_SO_Labse_NN-10_U_SKV/MW24_train_U_O.tsv'


    # Load test and train numpy files
    test_numpy = np.load(test_numpy_file_path)
    train_numpy = np.load(train_numpy_file_path)

    # Load test and train TSV files
    with open (test_TSV_file_path, 'r') as test_file:
        test_lines = [line.strip() for line in test_file]
    with open (train_TSV_file_path, 'r') as train_file:
        train_lines = [line.strip() for line in train_file]

    # Loop through test TSV and test embeddings
    for i, (test_embedding, test_line) in enumerate(tqdm.tqdm(zip(test_numpy, test_lines), total=len(test_lines))):
        # Loop through
        # if i == 20:  # For debugging limit to 10 iterations
        #     break

        all_nearest_sentences_list = []
        picked_train_file_lines = []

        if len(test_embedding.shape) == 1:
            test_embedding = test_embedding.reshape(1, -1)

        cosine_sim = cosine_similarity(test_embedding, train_numpy)
        top_x_sim_indices = np.argsort(cosine_sim[0])[::-1][:NNcount]   # top NNcount similar picks.

        for idx in top_x_sim_indices:
            picked_train_file_lines.append(train_lines[idx])

        test_fname, test_sntce, dom, slooot = finding_test_labels(test_line)

        limit_checked_picked_lines, limit_checked_picked_lines_len = check_context_token_limit(picked_train_file_lines, test_line, args)
        reversed_line_list = list(reversed(limit_checked_picked_lines))

        fin_sentence = ''
        for picked_line in reversed_line_list:
            fname, sentence, domains, slots = finding_test_labels(picked_line)
            conversation_in_format = ''
            domains = ast.literal_eval(domains)
            slots = ast.literal_eval(slots)
            new_dict = {domain: slot for domain, slot in zip(domains, slots)}
            
            if dialog_side == 'U':
                conversation_in_format = f'User: {sentence} Domain: ' + json.dumps(domains) + ' Slots: ' + json.dumps(new_dict) + '\n'
            
            elif dialog_side == 'UA':
                line_data = picked_line.split('\t')
                utterances = line_data[1].strip()
                utterances = utterances.split('---')
                for i, utterance in enumerate(utterances):
                    if i % 2 == 0:
                        user_utterance = utterance
                        conversation_in_format = conversation_in_format + "User: " + user_utterance.strip() 
                    else:
                        agent_utterance = utterance
                        conversation_in_format = conversation_in_format.strip() + " Agent: " + agent_utterance.strip() + ' '
                conversation_in_format = conversation_in_format.strip() + ' Domain: ' + json.dumps(domains) + ' Slots: ' + json.dumps(new_dict) + '\n'

            conversation_in_format = conversation_in_format.replace('  ', ' ')
            
            fin_sentence += conversation_in_format
            # print(f"{fin_sentence}")
        # breakpoint()
        # continue


        # add test sentence to the end of the list
        doma = ast.literal_eval(dom)
        if dialog_side == 'U':
            fin_sentence += 'User: ' + test_sntce + ' Domain: ' + json.dumps(doma) + ' Slots: {'  
        elif dialog_side == 'UA':
            test_sentence_processing = test_sntce.split('---')
            for i, sentence in enumerate(test_sentence_processing):
                if i % 2 == 0:
                    user_sentence = sentence
                    fin_sentence += 'User: ' + user_sentence
                else:
                    agent_sentence = sentence
                    fin_sentence += ' Agent: ' + agent_sentence + ' '
            fin_sentence = fin_sentence.strip() + ' Domain: ' + json.dumps(doma) + ' Slots: {'

        all_nearest_sentences_list.append(fin_sentence)
        
        # if there are double or more spaces in all_nearest_sentences_list, replace them with single space
        all_nearest_sentences_list = [sentence.replace('  ', ' ') for sentence in all_nearest_sentences_list]
        all_nearest_sentences_list = [sentence.replace('  ', ' ') for sentence in all_nearest_sentences_list]

    # Write each sentence to a new line in the file
        with open(output_file_name, 'a') as nn_write_file:
            nn_write_file.write(json.dumps(all_nearest_sentences_list) + '\n')

    print("Written to the file successfully")




    sleep(5)
    # Do th post processing here. 
    # 1. Removing the tags.
    if args.speaker_tag == 'N':
        # Remove User: Agent: speaker tags.
        # remove_speaker_tags
        processed_file_path = remove_speaker_tags(output_file_name, output_file_name)
    print("speaker_tags processed")
    sleep(5)

    # 2. slot_placeholder.
    if args.slot_placeholder != 'not mentioned':
        # replace the slot placeholder with argument value.
        # operate_slot_placeholder()
        slot_placeholder_output_file = operate_slot_placeholders(output_file_name, args.slot_placeholder, output_file_name)
        print(f"Processed file with slot placeholders saved at: {slot_placeholder_output_file}")
    print("slot placeholders processed")
    sleep(5)
    
    # 3. slot_key_sort.
    if args.slot_key_sort == 'Y' or args.slot_key_sort.isdigit():
        # sort the slots.
        # sort_slot_keys()
        sorted_shuffled_output_file = sort_and_shuffle_slot_keys(output_file_name, args.slot_key_sort, output_file_name)
        print(f"Processed file with sorted and shuffled slot keys saved at: {sorted_shuffled_output_file}")
    print("slot keys processed")
    sleep(10)

    print("Written to the file successfully")

if __name__ == "__main__":
    main()
