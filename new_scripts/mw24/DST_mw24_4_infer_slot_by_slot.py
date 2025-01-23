# sample run: 
# export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1); export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_EVALUATE_OFFLINE=1; python DST_mw24_4_infer_slot_by_slot.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import numpy as np
import argparse
import ast
import json
import os
import json_repair
import platform
from slot_infer_scripts.get_instruction_for_domain import get_instruction_for_domain
from slot_infer_scripts.append_model_name import append_model_name
from slot_infer_scripts.get_sorted_slots import get_sorted_slots

from TODx import finding_test_labels, remove_punctuations
a = torch.rand(1, 1).cuda()


# TODO: get proper instructions 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='allenai/OLMo-7B-Instruct', help='model name')  # allenai/OLMo-7B-Instruct | mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument('--output_folder_path', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/experiments/mw24/baseline/')
    parser.add_argument('--out_folder_prefix', type=str, default='DST_')

    parser.add_argument('--input_file', type=str,       default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MW24/baseline/MW24_OP_ST_PH-empty_SU_Labse_NN-3_U_SV/MW24_OP_ST_PH-empty_SU_Labse_NN-3_U_SV.txt', help='The input file to LLM')
    parser.add_argument('--instruction', type=int,      default=2, help='0=no instruction, 1=dynamic instruction, 2 = static instruction')

    parser.add_argument('--offset_start', type=int, default=0, help='Start offset for the test file')
    parser.add_argument('--offset_add', type=int, default=8000, help='Add offset for the test file')
    return parser.parse_args()

# output config file
config_string = '''
Dataset = "MW24"
Punct = original - "O" | no punctuation - "N" | model punctuation - "M"
Speaker_tag = "Y" | "N"
Slot_placeholder = "not mentioned" | "N.A." | "none" | deleting the slot key if there is no placeholder - "omit" | empty string - "empty"
Slot_key_sort = "Y" | "N" | seed - "1", "2" or number
Sentence_embedding_model = "sentence-transformers/LaBSE" 
NNcount = 10
Dialog_history = "Y" | "N"
Decoding = slot key and value given domain - "SKV" | slot value given slot key - "SV"
'''



# unsorted slots
dom2slots = {
    'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'], 
    'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'], 
    'attraction': ['type', 'name', 'area'], 
    'train': ['people', 'leaveAt', 'destination', 'day', 'arriveBy', 'departure'], 
    'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
}

# dom2slots = {
#     'taxi': ['arriveBy', 'departure', 'destination', 'leaveAt'], 
#     'restaurant': ['area', 'day', 'food', 'name', 'people', 'pricerange', 'time'], 
#     'attraction': ['area', 'name', 'type'], 
#     'train': ['arriveBy', 'day', 'departure', 'destination', 'leaveAt', 'people'], 
#     'hotel': ['area', 'day', 'internet', 'name', 'parking', 'people', 'pricerange', 'stars', 'stay', 'type']
# }


class NewlineStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, newline_token_id):
        self.tokenizer = tokenizer
        self.newline_token_id = newline_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the most recent token is a newline
        return input_ids[0, -1] == self.newline_token_id


    pass

def main():
    args = parse_args()

    # Create output folder
    output_folder_path = args.output_folder_path

    # Get basename of the input file without extension
    input_file_name = os.path.basename(args.input_file)
    input_file_name = os.path.splitext(input_file_name)[0]
    # If there is "SV" in the output_folder_name, replace it with "SKV"
    output_folder_name = input_file_name
    output_folder_name = append_model_name(output_folder_name, args.model_name)

    if "SKV" in input_file_name:
        output_folder_name = input_file_name.replace("SKV", "SV")

    # create the output folder if it does not exist. If it exists, print the message and exit
    output_folder_name = os.path.join(output_folder_path, output_folder_name)
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    # output log directory
    log_folder = os.path.join(output_folder_name, 'log_files')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # output log and config file name
    logg_file = os.path.join(log_folder, str(args.offset_start)+'_log.json')
    config_file = os.path.join(output_folder_name, str(args.offset_start)+'_config.json')

    # if the log file exists, print the message and exit
    if os.path.exists(logg_file):
        print(f"Output log file {logg_file} already exists. Exiting.")
        print("----------------")
        exit()

    # Get the test TSV file. Which is in the same folder as the input file. The name will have "test" and it is TSV file.
    input_data_folder = os.path.dirname(args.input_file)
    all_input_folder_files = os.listdir(input_data_folder)
    test_file = [file for file in all_input_folder_files if 'test' in file and '.tsv' in file][0]
    print(f"Test TSV file: {test_file}")

    # Load input file
    input_file = args.input_file
    with open(input_file, 'r') as f:
        in_file = f.readlines()

    # Load test file
    test_tsv_file_path = os.path.join(input_data_folder, test_file)
    with open(test_tsv_file_path, 'r') as tst_tsv_file:
        test_data = tst_tsv_file.readlines()

   # Create config file
    with open(config_file, 'w') as config_file:
        config_file.write(json.dumps(vars(args), indent=4))
        config_file.write(f'Output folder: {output_folder_name}')
        # Add details of the run
        config_file.write(platform.node())
        config_file.write('\n')
        config_file.write(platform.platform())
        config_file.write('\n')
        config_file.write(f"Output written at: {output_folder_name}")
        config_file.write('\n')
        config_file.write(config_string)

    # Model and tokenizer
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Get the token ID for newline
    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]




    counter = 0
    for i, (line, gold_line) in enumerate(zip(in_file, test_data)):
        if i < args.offset_start:
            counter += 1
            continue
        if i >= args.offset_start + args.offset_add:
            break
        counter += 1

        # get the gold line
        gold_fname, gold_sentence, gold_domain, gold_slots = finding_test_labels(gold_line)

        context = ast.literal_eval(line)[0]

        # get the context number
        context_len = context.count('\n')

        instruction = ''
        append_predicted = ''
        total_input_to_LLM = ''
        slot_history_dict = ''
        slot_history_dict_temp = ''
        slot_history_dict_fin = ''
        decoded_responses = []
        # print(gold_domain)
        for i, dom in enumerate(ast.literal_eval(gold_domain)):
            print(dom)
            # print(type(dom))
            if args.instruction == 0:
                instruction = ''
            elif args.instruction == 1: 
                instruction = get_instruction_for_domain(dom)
            elif args.instruction == 2:
                instruction = "Identify the slot value.\n"
            
            slot_history_dict = slot_history_dict_temp
            # print(slot_history_dict)
            # breakpoint()

            # Get the slot values in alphabetical order or based on seed number.
            slot_values = get_sorted_slots(dom, input_file_name)

            # slot_values = dom2slots[dom]
            if i == 0:
                slot_history_dict = '\"' + dom + '\": {'
            else:
                slot_history_dict += '}, \"' + dom + '\": {'

            for j, slot_key in enumerate(slot_values):
                # print(f"Slot value for {dom}: is {slot_value}")
                if j == 0:
                    slot_history_dict += '\"' + slot_key + '\":'
                else:
                    slot_history_dict += ', \"' + slot_key + '\":'
                
                total_input_to_LLM = instruction + context + slot_history_dict 
                # print(total_input_to_LLM)
                # print("------")

                input_ids = tokenizer.encode(total_input_to_LLM, return_tensors='pt')
                input_ids = input_ids.cuda()
                
                # Define stopping criteria
                stopping_criteria = StoppingCriteriaList([NewlineStoppingCriteria(tokenizer, newline_token_id)])

                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=12,
                        return_dict_in_generate=True,
                        output_scores=False,
                        stopping_criteria=stopping_criteria
                    )

                decoded_response = tokenizer.decode(output.sequences[0][input_ids.shape[-1]:], skip_special_tokens=True)
                decoded_responses.append(decoded_response)
                # print("------")
                # print(decoded_response)
                processed_output = decoded_response.split(',')[0].strip()
                processed_output = remove_punctuations([processed_output])[0].strip()
                # print("------")
                # print(processed_output)
                # val = "predicted value"
                val = processed_output



                # get the slot history
                slot_history_dict += ' \"' + val + '\"' 
                # breakpoint()

            # print(slot_history_dict)
            # slot_history_dict_fin += slot_history_dict
            slot_history_dict_temp = slot_history_dict 
            # print(slot_history_dict_temp)
            # breakpoint()
        slot_history_dict_final = '{' + slot_history_dict + '}}'
        print(slot_history_dict_final)
        print("======")
        # breakpoint()


        log_dictionary = {
            "line_number": counter,
            "gold_fname": gold_fname,
            "gold_domain": gold_domain,
            "gold_slots": gold_slots,
            "Instruction": instruction,
            "Context_length": context_len,
            "decoded_response": decoded_responses, 
            "processed_output": slot_history_dict_final,
            "total_input_to_LLM": total_input_to_LLM
            }

        with open(logg_file, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(log_dictionary))
            log_file.write('\n')

        

if __name__ == '__main__':
    main()

