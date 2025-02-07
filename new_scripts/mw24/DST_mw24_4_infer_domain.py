# Input file to the LLM and compute the domain metrics. 
# export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1); export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_EVALUATE_OFFLINE=1; python DST_mw24_4_infer_domain.py
import re
import json
from tqdm import tqdm
import os
import argparse
import numpy as np
import torch
import string
import ast
from tqdm import tqdm
from domain.find_x_similar_sentences import find_x_similar_sentences
from domain.finding_test_labels import finding_test_labels
from slot_infer_scripts.append_model_name import append_model_name
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from time import sleep


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_test_tsv', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MW24/punct/MW24_MP_ST_PH-nm_SO_Labse_NN-10_UA_SV/MW24_test_UA_MP.tsv', help='Path to the input file')
    parser.add_argument('--input_test_npy', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MW24/punct/MW24_MP_ST_PH-nm_SO_Labse_NN-10_UA_SV/MW24_test_UA_MP.npy', help='Path to the input file')
    parser.add_argument('--input_train_npy', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MW24/punct/MW24_MP_ST_PH-nm_SO_Labse_NN-10_UA_SV/MW24_train_UA_MP.npy', help='Path to the input file')
    parser.add_argument('--input_train_tsv', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MW24/punct/MW24_MP_ST_PH-nm_SO_Labse_NN-10_UA_SV/MW24_train_UA_MP.tsv', help='Path to the input file')

    parser.add_argument('--baseline_or_model', type=str, default='model', help='Whether to use the baseline or the model')
    parser.add_argument('--output_folder', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/experiments/MW24/domain', help='Path to the output file')
    parser.add_argument('--context_number', type=int, default=10, help='Number of similar sentences to find')

    parser.add_argument('--model_name', type=str, default='allenai/OLMo-7B-Instruct', help='Model name')
    parser.add_argument('--input_file_with_context', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MW24/punct/MW24_MP_ST_PH-nm_SO_Labse_NN-10_UA_SV.txt', help='Path to the input file')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for the input file')
    parser.add_argument('--offset_add', type=int, default=1000, help='End index for the input file')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')  # Add batch size
    return parser.parse_args()

def process_batch(model, tokenizer, instruction: str, batch: List[str]) -> List[str]:
    """Processes a batch of input strings and returns a list of decoded responses."""
    total_inputs_to_LLM = [instruction + input_str for input_str in batch]
    input_ids = tokenizer.batch_encode_plus(total_inputs_to_LLM, return_tensors='pt', padding=True, truncation=True, max_length=2048) # added truncation to prevent possible errors from the model
    input_ids = input_ids['input_ids'].cuda()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=12,
            return_dict_in_generate=True,
            output_scores=False
        )

    decoded_responses = tokenizer.batch_decode(output.sequences[:, input_ids.shape[-1]:], skip_special_tokens=True)
    return decoded_responses


def main():
    args = parse_arguments()
    train_numpy_file_path = args.input_train_npy
    test_numpy_file_path = args.input_test_npy
    input_test_file = args.input_test_tsv
    input_train_file = args.input_train_tsv
    context_number = args.context_number
    model_name = args.model_name
    baseline_or_model = args.baseline_or_model
    batch_size = args.batch_size

    os.makedirs(args.output_folder, exist_ok=True)

    # read the input test file.
    with open(input_test_file, 'r') as f:
        test_data = f.readlines()

    # read the numpy files.
    train_embeddings = np.load(train_numpy_file_path)
    test_embeddings = np.load(test_numpy_file_path)

    # read the training sentences once
    with open(input_train_file, 'r', encoding='utf-8') as f:
        train_sentences = [line.strip() for line in f]

    print("Entering loop")

    if baseline_or_model == 'baseline':

        output_file = os.path.join(args.output_folder, f"{os.path.basename(input_test_file).replace('.tsv', '')}_{context_number}_domain_predictions.tsv")

        for i, (test_line, test_embedding) in tqdm(enumerate(zip(test_data, test_embeddings))):
            gold_fname, gold_sentence, gold_domain, gold_slots = finding_test_labels(test_line)

            similar_sentences_lines = find_x_similar_sentences(train_embeddings, test_embedding, context_number, train_sentences)

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





    elif baseline_or_model == 'model':
        # create a folder of the input_file_with_context
        context_name = os.path.basename(args.input_file_with_context).replace('.txt', '')
        out_context_folder = os.path.join(args.output_folder, append_model_name(context_name, model_name))
        
        if not os.path.exists(out_context_folder):
            os.makedirs(out_context_folder)
        
        output_file_path = append_model_name(context_name, model_name) + f"_{args.start_index}_{args.offset_add}" + ".tsv"
        output_file_path_full = os.path.join(out_context_folder, output_file_path)

        print(f"Output will be saved to: {output_file_path_full}") # added print for debugging
        # sleep(10)


        legitmate_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']

        with open(args.input_file_with_context, 'r', encoding='utf-8') as f:
            with_context_lines = f.readlines()

        model_name = args.model_name
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
        model = model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set, especially important when padding is on the left
        instruction = 'Identify the domain(s) present in the conversation. The domains could be one or more of the following: [\'restaurant\', \'hotel\', \'attraction\', \'train\', \'taxi\']'

        # Prepare data for batch processing
        modified_context_lines = []
        gold_fnames = []
        gold_domains = []

        for gold_test_line, test_data_line in zip(test_data, with_context_lines):
            gold_fname, gold_sentence, gold_domain, gold_slots = finding_test_labels(gold_test_line)
            test_data_line_string = ast.literal_eval(test_data_line)[0]

            test_data_line_string_modified = test_data_line_string[:test_data_line_string.rfind("Domain: [") + len("Domain: [\"")]

            modified_context_lines.append(test_data_line_string_modified)
            gold_fnames.append(gold_fname)
            gold_domains.append(gold_domain)

        # Process in batches
        for i in tqdm(range(0, len(modified_context_lines), batch_size)):
            if i < args.start_index:
                # counter += 1
                print(f"Skipping index {i} because it's before start_index") # added print for debugging
                continue
            if i >= args.start_index + args.offset_add:
                print(f"Stopping at index {i} because it's beyond start_index + offset_add") # added print for debugging
                break
            
            batch_context = modified_context_lines[i:i + batch_size]
            batch_fnames = gold_fnames[i:i + batch_size]
            batch_domains = gold_domains[i:i + batch_size]

            # Get model responses for the batch
            decoded_responses = process_batch(model, tokenizer, instruction, batch_context)
            # print(decoded_responses)
            # print("-------------------")

            # Process the output and save to file
            for gold_fname, gold_domain, decoded_response in zip(batch_fnames, batch_domains, decoded_responses):
                processed_output = '["' + decoded_response.split(']')[0].strip() + ']'
                # if the processed output is not convertable to list, then consider it till the first blank space. If there is no blank space, then make it empty.
                try:
                    processed_output_list = ast.literal_eval(processed_output)  # Convert to a Python list
                except (SyntaxError, ValueError):  # Catch both SyntaxError and ValueError
                    processed_output_list = processed_output.split(' ')[0]
                    if not processed_output_list:
                        processed_output_list = []
                    else:
                        processed_output_list = [processed_output_list] # Convert back to list with one element

                # Delete the domain if it is not in the legitimate domains.
                processed_outpu_final = []
                if isinstance(processed_output_list, list): # Making sure processed_output_list is a list before iterating
                    for dom in processed_output_list:
                        if dom in legitmate_domains:
                            processed_outpu_final.append(dom)
                else:
                    print(f"Warning: processed_output_list is not a list: {processed_output_list}") # Log to see if it is ever a single string instead of list
                print("after processing:", processed_outpu_final) # added print for debugging
                with open(output_file_path_full, 'a', encoding='utf-8') as op_file: 
                    op_file.write(gold_fname + '\t' + gold_domain + '\t' + str(processed_outpu_final) + '\n')

if __name__ == '__main__':
    main()