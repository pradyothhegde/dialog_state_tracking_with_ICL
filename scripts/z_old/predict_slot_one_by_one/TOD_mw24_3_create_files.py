# export HF_HUB_OFFLINE=1 ;export HF_DATASETS_OFFLINE=1 ;export TRANSFORMERS_OFFLINE=1 ;export HF_EVALUATE_OFFLINE=1; export CUDA_VISIBLE_DEVICES=\$(free-gpus.sh 1); python debug_create_sim_sentences.py --start_index 0 --end_index 10 --output_file nearest_sentences.txt

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from tqdm import tqdm
import ast
from TODx import finding_test_labels, json_to_tsv_format, calculate_save_sentence_embedding
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=8000)
    parser.add_argument("--output_file", type=str, default='LLM_input_sentences.txt')
    parser.add_argument("--data_path", type=str, default='/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MULTIWOZ2.4/')
    parser.add_argument("--output_folder", type=str, default='/mnt/matylda4/hegde/int_ent/LLM_dialog_state/debug_model_out/TOD1_mw24_O7BI_U_P_10_DS_1_0')
    return parser.parse_args()

def main():
    # if the output folder does not exist, create it


    args = parse_args()
    data_path = args.data_path
    output_folder_path = args.output_folder
    input_test_data_path = os.path.join(data_path, 'mw24_DST_test.json')
    input_train_data_path = os.path.join(data_path, 'mw24_DST_train.json')

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # test_file_path = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/debug_model_out/TOD1_mw24_O7BI_U_P_10_DS_1_0/mw24_DST_test_U_punct_turns.tsv'
    # output_folder_path = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/debug_model_out/TOD1_mw24_O7BI_U_P_10_DS_1_0'

    # creating tsv and numpy files
    test_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_test_U_punct_turns.tsv')
    train_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_train_U_turns.tsv')
    test_numpy_file_path = os.path.join(output_folder_path, 'mw24_DST_test_U_punct_turns.npy')
    train_numpy_file_path = os.path.join(output_folder_path, 'mw24_DST_train_U_turns.npy')

    json_to_tsv_format(input_train_data_path, train_tsv_file_path, 'U')
    json_to_tsv_format(input_test_data_path, test_tsv_file_path, 'U')
    train_numpy_file_path = calculate_save_sentence_embedding(train_tsv_file_path, 'U', punct='punct')
    test_numpy_file_path = calculate_save_sentence_embedding(test_tsv_file_path, 'U', punct='punct')


    context_number = 10

    start_index = args.start_index
    end_index = args.end_index
    test_file_path = os.path.join(output_folder_path, 'mw24_DST_test_U_punct_turns.tsv')
    test_numpy_path = os.path.join(output_folder_path, 'mw24_DST_test_U_punct_turns.npy') 
    train_file_path = os.path.join(output_folder_path, 'mw24_DST_train_U_turns.tsv')
    train_numpy_path = os.path.join(output_folder_path, 'mw24_DST_train_U_turns.npy')

    test_numpy = np.load(test_numpy_path)
    train_numpy = np.load(train_numpy_path)
    
    nearest_neighbour_file_name = os.path.join(output_folder_path, args.output_file)
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-Instruct", trust_remote_code=True)
    
    with open(test_file_path, 'r') as test_file:
        test_lines = [line.strip() for line in test_file]
    with open(train_file_path, 'r') as train_file:
        train_lines = [line.strip() for line in train_file]

    def token_limit_checker(top_sentences, test_sentence):
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

            # change the token_checker to include the domain and slots
            temp_fname, temp_sentence, temp_domain, temp_slots = finding_test_labels(line)
            # temp_domain = ast.literal_eval(line.split('\t')[2])
            # temp_slots = ast.literal_eval(line.split('\t')[3])
            conversation = 'User: ' + temp_sentence  # Sentence for token count
            # print(f"temp conversation: {conversation}")
            # print(f"temp_domain: {temp_domain}")
            # print(f"temp_slots: {temp_slots}")

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
        # print(f"Test line: {test_line}")    #SNG0073.json	I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.	['taxi']	[{'leaveAt': 'N.A.', 'destination': 'pizza hut fenditton', 'departure': 'saint johns college', 'arriveBy': 'N.A.'}]
        # breakpoint()

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
        # in_format = f'User: {sentence} Domain: ["' + '", "'.join(domains) + '"] Slots: ' + json.dumps(new_dict) + '\n'

        limit_checked_picked_lines, limit_checked_picked_lines_len = token_limit_checker(picked_train_file_lines, test_line)
        reversed_line_list = list(reversed(limit_checked_picked_lines))
        # print(f" len(picked_train_file_lines): {len(picked_train_file_lines)}")
        # print(f" len(limit_checked_picked_lines): {len(limit_checked_picked_lines)}")

        fin_sentence = ''
        for picked_line in reversed_line_list:
            fname, sentence, domains, slots = finding_test_labels(picked_line)
            domains = ast.literal_eval(domains)
            slots = ast.literal_eval(slots)
            new_dict = {domain: slot for domain, slot in zip(domains, slots)}
            in_format = f'User: {sentence} Domain: ' + json.dumps(domains) + ' Slots: ' + json.dumps(new_dict) + '\n'
            
            in_format = in_format.replace('  ', ' ')
            
            fin_sentence += in_format

        # add test sentence to the end of the list
        

        doma = ast.literal_eval(dom)
        fin_sentence += 'User: ' + test_sntce + ' Domain: ' + json.dumps(doma) + ' Slots: {'  
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























