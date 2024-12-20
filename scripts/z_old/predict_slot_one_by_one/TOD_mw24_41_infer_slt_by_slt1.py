# export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1); export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_EVALUATE_OFFLINE=1; python load_and_infer_llm.py

import re
import json
import ast
from tqdm import tqdm
import os
import argparse
import numpy as np
import gc
import torch
a = torch.rand(1, 1).cuda()
import string
from debug_TODx import remove_punctuations, prepare_data_for_sentence_embedding, calculate_save_sentence_embedding, json_to_tsv_format
from debug_TODx import get_text_dom_in_template1, get_text_dom_in_template2, get_text_dom_slots_in_template1, get_text_dom_slots_in_template2, get_instruction_string 
from debug_TODx import finding_test_labels, find_x_similar_sentences, get_in_simple_user_agent_format, naming_function, get_text_dom_slots_in_template3
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import platform
import json_repair

device = torch.device('cuda')

# write to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Domain and Slot Prediction using LLM')
    parser.add_argument('--model', type=str, default='allenai/OLMo-7B-Instruct', help='Model name')
    parser.add_argument('--data_path', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/Data/MULTIWOZ2.4/',  help='Path to the data directory')
    parser.add_argument('--out_dir', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments', help='Path to the LLM output directory')
    parser.add_argument('--dialog_side', default='U', type=str, help='Calculate sentence embedding option: U - for user only sentences, UA - for user-agent sentences')
    parser.add_argument('--punct', default='punct', type=str, help='Remove punctuations from the sentences.: punct - with punctuations, no_punct - without punctuations')   
    parser.add_argument('--context_number', type=int, default=10, help='Number of similar sentences to find in the training set')
    parser.add_argument('--predict', type=str, default='domain_slots', help='Predict domain or slots: domain - predict domain, domain_slots - predict slots')
    parser.add_argument('--input_template', type=int, default=1, help='Input format for LLM: 1, 2, 3... Check the format readme file')
    parser.add_argument('--instruction_preset', type=int, default=0, help='Instruction template for LLM: 0, 1 Check the format readme file')
    # parser.add_argument('--skip_computing_embeddings', default=True, help='Skip computing embeddings and TSV files')
    parser.add_argument('--offset_start', type=int, default=0, help='Start offset for the test file')
    parser.add_argument('--offset_add', type=int, default=8000, help='Add offset for the test file')
    parser.add_argument('--lowercase_input', default=False, help='Lowercase the input to the LLM')
    return parser.parse_args()

dom2slots = {
    'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'], 
    'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'], 
    'attraction': ['type', 'name', 'area'], 
    'train': ['people', 'leaveAt', 'destination', 'day', 'arriveBy', 'departure'], 
    'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
}

def get_instruction_for_domain(domain):
    domain_processing_string = "[" + domain + "]"
    # domain_list = ast.literal_eval(domain_processing_string)
    print(type(domain))
    # print(domains) # domains is a list
    # build instruction
    # Instruction: Identify slot names and values from the dialogue. The domain is taxi. The slots for taxi are leaveAt, destination, departure, arriveBy.
    instruction = "Identify slot names and values from the dialogue. The domain is " + domain + "." + " The slots for " + domain + " are "
    for j in range(len(dom2slots[domain])):
        instruction += dom2slots[domain][j]
        if j != len(dom2slots[domain]) - 1:
            instruction += ", "
        else:
            instruction += "."
    return instruction


def main():
    print("start")
    args = parse_arguments()

    counter = args.offset_start
    global_hits = 0
    global_misses = 0

    max_new_tokenss = 0
    if args.predict == 'domain':
        print('Predicting domain')
        max_new_tokenss = 13
    elif args.predict == 'domain_slots':
        print('Predicting slots')
        max_new_tokenss = 10

    offset_start = args.offset_start
    offset_add = args.offset_add

    dom2slots = {
    'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'], 
    'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'], 
    'attraction': ['type', 'name', 'area'], 
    'train': ['people', 'leaveAt', 'destination', 'day', 'arriveBy', 'departure'], 
    'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
}

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # output folder path
    model_name, punct_name, predicting = naming_function(args.model, args.punct, args.predict)
    dataset_abbrv = 'mw24'  # change that name of the dataset.

    op_folder_name = 'TOD1_' + dataset_abbrv + '_' + model_name + '_' + str(args.dialog_side) + '_' + punct_name + '_' + str(args.context_number) + '_' + predicting + '_' + str(args.input_template) + '_' + str(args.instruction_preset)
    output_folder_path = os.path.join(args.out_dir, op_folder_name)
    os.makedirs(output_folder_path, exist_ok=True)

    # output config file
    config_file = os.path.join(output_folder_path, op_folder_name + '_config.json')
    with open(config_file, 'w') as config_file:
        config_file.write(json.dumps(vars(args), indent=4))
        # Add details of the run
        config_file.write(platform.node())
        config_file.write('\n')
        config_file.write(platform.platform())
        config_file.write('\n')

    # Load test file
    test_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_test_U_punct_turns.tsv')
    with open(test_tsv_file_path, 'r') as tst_tsv_file:
        test_data = tst_tsv_file.readlines()
    # +++++++++++++++++++++++++++++++++++++++++++++++++
    # exit(0)
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    similar_sentences_file_path = os.path.join(output_folder_path, 'LLM_input_sentences.txt')
    with open(similar_sentences_file_path, 'r') as sim_file:
        sim_stored_lines = sim_file.readlines()
        sim_stored_lines = [line.strip() for line in sim_stored_lines]

    # output_file = os.path.join(output_folder_path, op_folder_name + '_' + str(offset_start)+ '_' + str(offset_add) +'_results.tsv')
    # output_file = os.path.join(output_folder_path, op_folder_name +'_results.tsv')

    # # Load model and tokenizer
    # accelerator = Accelerator()

    # if args.model:
    #     tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    #     model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    # else:
    #     print("model is necessary")

    # # Prepare model and tokenizer with accelerator
    # model, tokenizer = accelerator.prepare(model, tokenizer)
    # model.eval()
    
    # log file folder
    log_folder = os.path.join(output_folder_path, 'log_files')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # log file name
    logg_file = os.path.join(output_folder_path, 'log_files', op_folder_name + '_' + str(args.offset_start) + '_log.json')
    
    print("entering loop")
    # breakpoint()
    counter = offset_start
    total_domain_count = 0
    total_slot_count = 0
    slot_count = 0
    for i, test_line in enumerate(test_data[offset_start:offset_start+offset_add+1]):
    # for i, (test_line, sim_sentences_line) in enumerate(zip(test_data, sim_stored_lines)):
        print("looping")
        
        decoded_response = ''
        # find all gold test labels: test_sentence, domain, slots
        gold_fname, gold_sentence, gold_domain, gold_slots = finding_test_labels(test_line)
        total_domain_count += len(ast.literal_eval(gold_domain))
        instruction_string = ''
        # instruction_string = get_instruction_string(args.instruction_preset, args.predict)

        # Prepare input for LLM

        context = ''
        slot_dict = {}
        decoded_slot_dict = {}
        final_slot_dict = {}
        final_decoded_dict = {}
        hits = 0
        misses = 0
        slot_count = 0
        total_input_to_LLM =   ''


        

        for i, (loop_domain, loop_slots) in enumerate(zip(ast.literal_eval(gold_domain), ast.literal_eval(gold_slots))):
            total_domain_count += 1
            input_to_LLM = json.loads(sim_stored_lines[counter])[0]
            input_to_LLM = input_to_LLM + '"' + loop_domain + '": {"' 
            print(input_to_LLM)
            breakpoint()
            continue
            for slots in loop_slots:
                # print(f"Domain: {loop_domain}, Slot: {slots}")
                total_slot_count += 1
                slot_count += 1
                # Frame total input to LLM
                input_to_LLM = json.loads(sim_stored_lines[counter])[0]
                # print(input_to_LLM)
                # breakpoint()
                # continue
                total_input_to_LLM = input_to_LLM + '"' + loop_domain + '": {"' + slots + '":'
                # print(total_input_to_LLM)
                # breakpoint()
                # continue
                with torch.no_grad():
                    tok_inp = tokenizer.encode(total_input_to_LLM, add_special_tokens=False, return_tensors='pt')
                    tok_inp = tok_inp.to(device=device)
                    decoded_response = model.generate( 
                        input_ids=tok_inp,              
                        max_new_tokens=max_new_tokenss,
                        return_dict_in_generate=True,
                        output_scores=False
                    )
                    decoded_response = tokenizer.decode(decoded_response.sequences[0][tok_inp.shape[-1]:], skip_special_tokens=True)
                #     print("* decoded response:", decoded_response)
                # print("====================================================")
                prediction_to_append = decoded_response.strip().replace('\n', ' ').replace('\t', ' ')
                slot_prediction = prediction_to_append.split(",")[0]
                prediction_to_append = remove_punctuations([slot_prediction])[0]
                slot_dict[slots] = prediction_to_append
                decoded_slot_dict[slots] = decoded_response
                # print(slot_dict)
                # print(decoded_slot_dict)

                # check if hit or miss
                gold_label = remove_punctuations([loop_slots[slots]])[0]   # remove punctuations
                # print(f"Gold: {gold_label}, Prediction: {prediction_to_append}")
                if prediction_to_append == gold_label:
                        hits += 1
                else:
                    misses += 1
                # add slot_dict to the final_dict
                # breakpoint()
            final_decoded_dict[loop_domain] = decoded_slot_dict
            final_slot_dict[loop_domain] = slot_dict
        counter += 1

        # Write to the output file
        log_dictionary = {
            "line_number": counter,
            "gold_fname": gold_fname,
            "gold_domain": gold_domain,
            "gold_slots": gold_slots,
            "decoded_response": final_decoded_dict, 
            "processed_output": final_slot_dict,
            "hits":  hits,
            "misses": misses,
            "total hits and misses = total number of slots" : hits+misses,
            "number of slots present": slot_count,
            # add number of contexts
            "total_input_to_LLM": total_input_to_LLM
            }
        
        with open(logg_file, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(log_dictionary))
            # log_file.write(json.dumps(log_dictionary, indent=4))
            log_file.write('\n')
        # breakpoint()
        
        # gc.collect()
        # torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


