# This script is to create the "input sentences" for LLM model.
#@@@ In detail @@@#
# 1. Load the test and train data
# 2. Create the tsv files for test and train data: which will be used to create the numpy files
# 3. Create the numpy files for test and train data: which will be used to calculate the cosine similarity
# 4. Load the numpy files
# 5. Calculate the cosine similarity between the test and train data
# 6. Pick the top 10 similar sentences from the train data. If the total token length exceeds 2048, then limit the number of sentences 
# 7. Create the input sentences for LLM model, remove punctuation of utterances in the process if required. 
#@@@ Sample command @@@#
# export HF_DATASETS_OFFLINE=1 ;export TRANSFORMERS_OFFLINE=1 ;export HF_EVALUATE_OFFLINE=1; export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1); python TOD_mw24_3_create_files.py

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from tqdm import tqdm
import ast
from TODx import finding_test_labels, json_to_tsv_format, calculate_save_sentence_embedding, remove_punctuations
import json
import argparse
import torch
# a = torch.randn(1,1).cuda()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=8000)
    # parser.add_argument("--output_file", type=str, default='LLM_input_sentences_U.txt')
    parser.add_argument("--data_path", type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/Data/MULTIWOZ2.4')
    parser.add_argument("--output_folder", type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/NecessaryFiles/')
    parser.add_argument("--model_name", type=str, default='allenai/OLMo-7B-Instruct')
    parser.add_argument("--context_number", type=int, default=10)
    parser.add_argument("--dialog_side", type=str, default='U', help="U=User, UA=User-Agent")
    parser.add_argument("--tsv_npy_creation", type=str, default='Y', help="Y=yes, N=no")
    parser.add_argument("--punct", type=str, default='Y', help="Y=yes, N=no")
    return parser.parse_args()

def main():

    args = parse_args()
    data_path = args.data_path
    output_folder_path = args.output_folder
    context_number = args.context_number
    model_name = args.model_name
    dialog_side = args.dialog_side

    # if output folder does not exist, create it
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    input_test_data_path = os.path.join(data_path, 'mw24_DST_test.json')
    input_train_data_path = os.path.join(data_path, 'mw24_DST_train.json')
    input_val_data_path = os.path.join(data_path, 'mw24_DST_valid.json')

    # creating tsv and numpy files
    if args.dialog_side == 'U':
        nearest_neighbour_file_name = os.path.join(output_folder_path, 'LLM_input_sentences_U.txt')
        if args.punct == 'N':
            nearest_neighbour_file_name = os.path.join(output_folder_path, 'LLM_input_sentences_U_PN.txt')

        test_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_test_U_punct_turns.tsv')
        train_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_train_U_turns.tsv')
        valid_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_val_U_turns.tsv')
        test_numpy_file_path = os.path.join(output_folder_path, 'mw24_DST_test_U_punct_turns.npy')
        train_numpy_file_path = os.path.join(output_folder_path, 'mw24_DST_train_U_turns.npy')

        if args.tsv_npy_creation == 'Y':
            json_to_tsv_format(input_train_data_path, train_tsv_file_path, 'U')
            json_to_tsv_format(input_test_data_path, test_tsv_file_path, 'U')
            # json_to_tsv_format(input_val_data_path, valid_tsv_file_path, 'U')
            train_numpy_file_path = calculate_save_sentence_embedding(train_tsv_file_path, 'U', punct='punct')
            test_numpy_file_path = calculate_save_sentence_embedding(test_tsv_file_path, 'U', punct='punct')
    
    elif args.dialog_side == 'UA':
        nearest_neighbour_file_name = os.path.join(output_folder_path, 'LLM_input_sentences_UA.txt')
        if args.punct == 'N':
            nearest_neighbour_file_name = os.path.join(output_folder_path, 'LLM_input_sentences_UA_PN.txt')

        test_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_test_UA_punct_turns.tsv')
        train_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_train_UA_turns.tsv')
        valid_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_val_UA_turns.tsv')
        test_numpy_file_path = os.path.join(output_folder_path, 'mw24_DST_test_UA_punct_turns.npy')
        train_numpy_file_path = os.path.join(output_folder_path, 'mw24_DST_train_UA_turns.npy')

        if args.tsv_npy_creation == 'Y':
            json_to_tsv_format(input_train_data_path, train_tsv_file_path, 'UA')
            json_to_tsv_format(input_test_data_path, test_tsv_file_path, 'UA')
            # json_to_tsv_format(input_val_data_path, valid_tsv_file_path, 'UA')
            train_numpy_file_path = calculate_save_sentence_embedding(train_tsv_file_path, 'UA', punct='punct')
            test_numpy_file_path = calculate_save_sentence_embedding(test_tsv_file_path, 'UA', punct='punct')

    print("Created the tsv and numpy files successfully")
    # exit()

    # loading the numpy files
    start_index = args.start_index
    end_index = args.end_index
    if args.dialog_side == 'U':
        test_file_path = os.path.join(output_folder_path, 'mw24_DST_test_U_punct_turns.tsv')
        test_numpy_path = os.path.join(output_folder_path, 'mw24_DST_test_U_punct_turns.npy') 
        train_file_path = os.path.join(output_folder_path, 'mw24_DST_train_U_turns.tsv')
        train_numpy_path = os.path.join(output_folder_path, 'mw24_DST_train_U_turns.npy')
    elif args.dialog_side == 'UA':
        test_file_path = os.path.join(output_folder_path, 'mw24_DST_test_UA_punct_turns.tsv')
        test_numpy_path = os.path.join(output_folder_path, 'mw24_DST_test_UA_punct_turns.npy') 
        train_file_path = os.path.join(output_folder_path, 'mw24_DST_train_UA_turns.tsv')
        train_numpy_path = os.path.join(output_folder_path, 'mw24_DST_train_UA_turns.npy')

    test_numpy = np.load(test_numpy_path)
    train_numpy = np.load(train_numpy_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    with open(test_file_path, 'r') as test_file:
        test_lines = [line.strip() for line in test_file]
    with open(train_file_path, 'r') as train_file:
        train_lines = [line.strip() for line in train_file]

    def token_limit_checker(top_sentences, test_sentence, dialog_side):
        limited_top_sentences = []
        total_tok_len = 0
        tmp_tst_fname, tmp_tst_sentence, tmp_tst_domain, tmp_tst_slots = finding_test_labels(test_sentence)
        tmp_tst_conv = 'User: ' + tmp_tst_sentence
        tmp_tsv_domain = ast.literal_eval(tmp_tst_domain)
        tmp_tsv_final_slots = {}
        for tmp_dom, tmp_slt in zip(ast.literal_eval(tmp_tst_domain), ast.literal_eval(tmp_tst_slots)):
            tmp_tsv_final_slots[tmp_dom] = tmp_slt
        tst_token_checker = tmp_tst_conv + ' Domain: ["' + '", "'.join(tmp_tsv_domain) + '"] Slots: ' + str(tmp_tsv_final_slots)
        test_tokens = tokenizer(tst_token_checker, return_tensors='pt')
        test_tok_len = test_tokens['input_ids'].shape[1]
        max_tokens = 2048 - 64 - test_tok_len

        for line in top_sentences:
            conversation = ''
            _, _, temp_domain, temp_slots = finding_test_labels(line)
            if dialog_side == 'U':
                # change the token_checker to include the domain and slots
                temp_fname, temp_sentence, temp_domain, temp_slots = finding_test_labels(line)
                # temp_domain = ast.literal_eval(line.split('\t')[2])
                # temp_slots = ast.literal_eval(line.split('\t')[3])
                if args.punct == 'N':
                    temp_sentence = remove_punctuations([temp_sentence])[0]
                conversation = 'User: ' + temp_sentence  # Sentence for token count
                # print(f"temp conversation: {conversation}")
                # print(f"temp_domain: {temp_domain}")
                # print(f"temp_slots: {temp_slots}")
            elif dialog_side == 'UA':
                line_data = line.split('\t')
                utterances = line_data[1].strip()
                utterances = utterances.split('---')
                # print(f"utterances: {utterances}")
                # print(f"type of utterances: {type(utterances)}")
                for i, utterance in enumerate(utterances):
                    if args.punct == 'N':
                        utterance = remove_punctuations([utterance])[0]
                    if i % 2 == 0:
                        user_utterance = utterance
                        conversation = conversation + "User: " + user_utterance.strip() 
                    else:
                        agent_utterance = utterance
                        conversation = conversation.strip() + " Agent: " + agent_utterance.strip() + ' '
                # print(f"conversation: {conversation}")
            # breakpoint()
            # continue
            
            slot_dict = {}
            for tmp_dom, tmp_slt in zip(ast.literal_eval(temp_domain), ast.literal_eval(temp_slots)):
                # print(f"domain: {tmp_dom}, slot: {tmp_slt}")
                slot_dict[tmp_dom] = tmp_slt
            token_checker = conversation + ' Domain: ["' + '", "'.join(ast.literal_eval(temp_domain)) + '"] Slots: ' + str(slot_dict)
            
            # print(f"token_checker: {token_checker}")            
            # breakpoint()
            tok_line = tokenizer(token_checker, return_tensors='pt')
            tok_len = tok_line['input_ids'].shape[1]
            total_tok_len += tok_len

            if total_tok_len <= max_tokens:
                limited_top_sentences.append(line)
            else:
                break

        return limited_top_sentences, len(limited_top_sentences)

    for i, (test_embedding, test_line) in enumerate(tqdm(zip(test_numpy, test_lines), total=len(test_lines))):
        if i < start_index:
            continue
        elif i > end_index:
            break
        # print(f"Test line: {test_line}")   

        all_nearest_sentences_list = []
        picked_train_file_lines = []

        if len(test_embedding.shape) == 1:
            test_embedding = test_embedding.reshape(1, -1)

        cosine_sim = cosine_similarity(test_embedding, train_numpy)
        top_x_sim_indices = np.argsort(cosine_sim[0])[::-1][:context_number]
        
        for idx in top_x_sim_indices:
            picked_train_file_lines.append(train_lines[idx])

        # gold_fname, gold_sentence, gold_domain, gold_slots = finding_test_labels(test_line)
        test_fname, test_sntce, dom, slooot = finding_test_labels(test_line)
        # print(f"Test sentence: {test_sntce}")    
        # conversation_in_format = f'User: {sentence} Domain: ["' + '", "'.join(domains) + '"] Slots: ' + json.dumps(new_dict) + '\n'

        limit_checked_picked_lines, limit_checked_picked_lines_len = token_limit_checker(picked_train_file_lines, test_line, dialog_side)
        reversed_line_list = list(reversed(limit_checked_picked_lines))
        # print(f" len(picked_train_file_lines): {len(picked_train_file_lines)}")
        # print(f" len(limit_checked_picked_lines): {len(limit_checked_picked_lines)}")

        fin_sentence = ''
        for picked_line in reversed_line_list:
            fname, sentence, domains, slots = finding_test_labels(picked_line)
            conversation_in_format = ''
            domains = ast.literal_eval(domains)
            slots = ast.literal_eval(slots)
            new_dict = {domain: slot for domain, slot in zip(domains, slots)}
            
            if dialog_side == 'U':
                if args.punct == 'N':
                    sentence = remove_punctuations([sentence])[0]
                conversation_in_format = f'User: {sentence} Domain: ' + json.dumps(domains) + ' Slots: ' + json.dumps(new_dict) + '\n'
            
            elif dialog_side == 'UA':
                line_data = picked_line.split('\t')
                utterances = line_data[1].strip()
                utterances = utterances.split('---')
                for i, utterance in enumerate(utterances):
                    if args.punct == 'N':
                        utterance = remove_punctuations([utterance])[0]
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
            if args.punct == 'N':
                test_sntce = remove_punctuations([test_sntce])[0]
            fin_sentence += 'User: ' + test_sntce + ' Domain: ' + json.dumps(doma) + ' Slots: {'  
        elif dialog_side == 'UA':
            test_sentence_processing = test_sntce.split('---')
            for i, sentence in enumerate(test_sentence_processing):
                if args.punct == 'N':
                    sentence = remove_punctuations([sentence])[0]
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

        
        # if i == 10:  # For debugging limit to 10 iterations
        #     break

    # Write each sentence to a new line in the file
        with open(nearest_neighbour_file_name, 'a') as nn_write_file:
            nn_write_file.write(json.dumps(all_nearest_sentences_list) + '\n')

    print("Written to the file successfully")
    

if __name__ == '__main__': 
    main()
    print("This program attained Mukti")























