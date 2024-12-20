# sample run: 
# export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1); export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_EVALUATE_OFFLINE=1; python TOD_open_ended_domain_wise.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import numpy as np
import argparse
import ast
import json
import os
import json_repair
import platform

from TODx import finding_test_labels
a = torch.rand(1, 1).cuda()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='allenai/OLMo-7B-Instruct', help='model name')    # allenai/OLMo-7B-Instruct | mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument('--nec_files', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/NecessaryFiles/')
    parser.add_argument('--output_folder_path', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/mw24/O7BI/')
    parser.add_argument('--out_folder_prefix', type=str, default='TOD_')
    parser.add_argument('--context_number', type=int, default=10)

    parser.add_argument('--input_file', type=str,       default='LLM_input_U_PY_TY_agent_added_n_limited_nn_rem_NA.txt', help='Check necessary files folder for the appropriate file')
    parser.add_argument('--dialog_side', type=str,      default='U', help='U=user, UA=user-agent')
    parser.add_argument('--punct', type=str,            default='Y', help='Y=yes, N=no')
    parser.add_argument('--tag', type=str,              default='Y', help='Y=yes, N=no')
    parser.add_argument('--instruction', type=int,      default=1, help='0=no instruction, 1=dynamic instruction, 2 = static instruction')

    parser.add_argument('--offset_start', type=int, default=0, help='Start offset for the test file')
    parser.add_argument('--offset_add', type=int, default=8000, help='Add offset for the test file')
    return parser.parse_args()

# output config file
config_string = '''
This is the configuration file for the run.
TOD_mw24_O7BI_SD_I1_10_PY_TY filename indicates the following:
TOD - Task of Dialogue
mw24 - MultiWOZ 2.4 dataset
O7BI - OLMo 7B Instruct model
SD - Slot Detection based on domain (domain wise, open ended), SS - Slot Detection based on slot by slot
I1 - 0 = No instruction, Instruction type 1 (Dynamic instruction), 2 = Static instruction
10 - Maximum context number
PY - Punctuation Yes
TY - Tagging Yes
This is extracted from the input file.
'''


# args.dialog_side

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
    # print(type(domain))
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

    # pass

def get_instruction(process_line):
    # find domain
    # print("Process line: ", process_line)
    process_line = ast.literal_eval(process_line)[0]
    # Domain: ["taxi"] Slots: {
    temp_text = process_line.split('\n')
    # get the last element in temp_text
    temp_text = temp_text[-1]
    templ_domain = temp_text.split('Domain: ')
    # get things within [ ]
    domain = templ_domain[1].split('[')[1].split(']')[0]
    # print("Domain: ", domain)
    # print("Last line: ", temp_text)
    domain_process_string = "[" + domain + "]"
    domains = ast.literal_eval(domain_process_string)
    # print(domains) # domains is a list

    # build instruction
    # Instruction: Identify slot names and values from the dialogue. The domains are taxi, restaurant. The slots for taxi are leaveAt, destination, departure, arriveBy. The slots for restaurant are people, day, time, food, pricerange, name, area.
    # if there is only one domain, then the instruction will the "The domain is taxi":
    if len(domains) == 1:
        instruction = "Identify slot names and values from the dialogue. The domain is "
    else:
        instruction = "Identify slot names and values from the dialogue. The domains are "
    for i in range(len(domains)):
        instruction += domains[i]
        if i != len(domains) - 1:
            instruction += ", "
        else:
            instruction += "."
    
    for i in range(len(domains)):
        instruction += " The slots for "
        instruction += domains[i] + " are "
        for j in range(len(dom2slots[domains[i]])):
            instruction += dom2slots[domains[i]][j]
            if j != len(dom2slots[domains[i]]) - 1:
                instruction += ", "
            else:
                instruction += "."
    # print(instruction)

    # breakpoint()
    # pass
    instruction = instruction + "\n" 
    return instruction

class NewlineStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, newline_token_id):
        self.tokenizer = tokenizer
        self.newline_token_id = newline_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the most recent token is a newline
        return input_ids[0, -1] == self.newline_token_id

def get_details_from_input_file(input_file_name):
    # input_file_name = 'LLM_input_sentences_UA_PN_rem_NA.txt'
    if '_UA' in input_file_name:
        dialog_side = 'UA'
    elif '_U' in input_file_name:
        dialog_side = 'U'
    
    if '_PN' in input_file_name:
        punct = 'N'
    else:
        punct = 'Y'

    if '_TN' in input_file_name:
        tag = 'N'
    else:
        tag = 'Y'

    return dialog_side, punct, tag



def main():
    args = parse_args()

    # Create output folder
    output_folder_path = args.output_folder_path

    if args.model_name == 'allenai/OLMo-7B-Instruct':
        model_name_short = 'O7BI'
    elif args.model_name == 'mistralai/Mistral-7B-Instruct-v0.3':
        model_name_short = 'M7BIV03'
    input_data_folder = args.nec_files
    context_number = args.context_number
    dialog_side = args.dialog_side
    punct = args.punct
    tag = args.tag
    dataset_name = 'mw24'
    arg_instruction = args.instruction
    
    input_file_name = args.input_file
    dialog_side, punct, tag = get_details_from_input_file(input_file_name)

    
    output_folder_name = args.out_folder_prefix + dataset_name + '_' + model_name_short + '_SD_I' + str(arg_instruction) + '_' + str(dialog_side) + '_' + str(context_number) + '_P' + str(punct) + '_T' + str(tag)
    output_folder_name = os.path.join(output_folder_path, output_folder_name)
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    # Model and tokenizer
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Get the token ID for newline
    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]


    # Create log file
    log_folder = os.path.join(output_folder_name, 'log_files')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    logg_file = os.path.join(log_folder, str(args.offset_start)+'_log.json')
    # if log file exists, print the message and exit
    if os.path.exists(logg_file):
        print(f"Log file {logg_file} exists. Exiting.")
        exit()

    # Load input file
    # input_file = args.input_file
    input_file = os.path.join(input_data_folder, args.input_file)
    with open(input_file, 'r') as f:
        in_file = f.readlines()
    
    # Load test file
    test_tsv_file_path = os.path.join(input_data_folder, 'mw24_DST_test_U_punct_turns.tsv')
    with open(test_tsv_file_path, 'r') as tst_tsv_file:
        test_data = tst_tsv_file.readlines()

    # Create config file
    config_file = os.path.join(output_folder_name, str(args.offset_start)+'_config.json')
    with open(config_file, 'w') as config_file:
        config_file.write(json.dumps(vars(args), indent=4))
        config_file.write('Paramenters extracted from the input file. (Need not consider from the args above, consider from below.)\n')
        config_file.write(f" dialog side: {dialog_side}, punctuation: {punct}, tag: {tag}")
        # Add details of the run
        config_file.write(platform.node())
        config_file.write('\n')
        config_file.write(platform.platform())
        config_file.write('\n')
        config_file.write(f"Output written at: {output_folder_name}")
        config_file.write('\n')
        config_file.write(config_string)

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

        # instruction = get_instruction(line)

        # if len(ast.literal_eval(gold_domain)) == 1:
        #     continue
        instruction = ''
        append_predicted = ''
        total_input_to_LLM = ''
        slot_history_dict = ''
        decoded_responses = []
        for i, dom in enumerate(ast.literal_eval(gold_domain)):
        # for i, (dom, g_slots) in enumerate(zip(ast.literal_eval(gold_domain), ast.literal_eval(gold_slots))):
            # print(dom)
            # print(type(dom))
            if args.instruction == 0:
                instruction = ''
            elif args.instruction == 1: 
                instruction = get_instruction_for_domain(dom)
            elif args.instruction == 2:
                instruction = "Identify the slots"

            if i == 0:
                total_input_to_LLM = instruction + '\n' + context + "\""+ dom + "\": {" 
                # print(total_input_to_LLM)
                # print(i)
            else:
                total_input_to_LLM = instruction + '\n' + context + slot_history_dict + ', \"' + dom + '\": {'
                # print(total_input_to_LLM)
                # print(i)
                if slot_history_dict:
                    slot_history_dict = slot_history_dict + ', '
            #     print("-------------------")
            # total_input_to_LLM = instruction + '\n' + context + "\""+ dom + "\": {"
            # print("------")
            # print(total_input_to_LLM)
            # print("------")

            input_ids = tokenizer.encode(total_input_to_LLM, return_tensors='pt')
            input_ids = input_ids.cuda()
            
            # Define stopping criteria
            stopping_criteria = StoppingCriteriaList([NewlineStoppingCriteria(tokenizer, newline_token_id)])

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    return_dict_in_generate=True,
                    output_scores=False,
                    stopping_criteria=stopping_criteria
                )

            decoded_response = tokenizer.decode(output.sequences[0][input_ids.shape[-1]:], skip_special_tokens=True)
            # print(decoded_response)
            decoded_responses.append(decoded_response)
            # trim decoded_response till '}'
            decoded_response = decoded_response.split('}')[0] + '}'
            decoded_plus = '{' + decoded_response
            processed_output = json_repair.loads(decoded_plus)
            # print("-------")
            # print(processed_output)
            append_predicted = json.dumps(processed_output)
            slot_history_dict = slot_history_dict + '\"' + dom + '\": ' + append_predicted 
        # write to file
            # print("========")
            
            # print(slot_history_dict)

        # continue
        slot_history_dict = '{' + slot_history_dict + '}'
        
        log_dictionary = {
            "line_number": counter,
            "gold_fname": gold_fname,
            "gold_domain": gold_domain,
            "gold_slots": gold_slots,
            "Instruction": instruction,
            "Context_length": context_len,
            "decoded_response": decoded_responses, 
            "processed_output": slot_history_dict,
            "total_input_to_LLM": total_input_to_LLM
            }

        with open(logg_file, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(log_dictionary))
            log_file.write('\n')
        # breakpoint()

        # # clear GPU memory
        # del input_ids
        # del output
        # del decoded_response
        # del processed_output
        # del append_predicted
        # del slot_history_dict
        # del log_dictionary
        # torch.cuda.empty_cache()
        
        

if __name__ == '__main__':
    main()

